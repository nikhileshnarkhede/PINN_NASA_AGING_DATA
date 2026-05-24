"""
data/features.py
================
Preprocessing pipeline and RUL label construction for the IGBT
collector-emitter voltage (Vce) signal.

Implements the three ordered preprocessing steps from
  Lu, Guo, Liu & Shi (2023), Scientific Reports 13:10167:

  1. average downsampling -- one sample per square-wave cycle
  2. standardization      -- zero mean, unit standard deviation
  3. EMA window smoothing -- exponential moving average (paper Eq. 14)

plus RUL label construction (RUL(t) = 1 - t / Nf, paper Eq. 1).

Leak-free standardization
-------------------------
For an honest out-of-sample (leave-one-device-out) test the test device must
not influence its own normalisation. The standardization step is therefore
SPLIT from the rest of the pipeline so the caller can:

  * downsample every device   -> downsample_only()        (no cross-device stats)
  * fit mu / sigma on TRAIN devices only -> fit_standardizer()
  * standardize + smooth every device with those fixed stats
                              -> standardize_and_smooth()

data/loader.py orchestrates exactly that. All functions take and return numpy
arrays; preprocessing parameters live in config/config.py.
"""
import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ============================================================================
# Step 1 -- average downsampling
# ============================================================================
def average_downsample(signal: np.ndarray, window: int) -> np.ndarray:
    """
    Average downsampling.

    The square-wave gate signal makes the raw Vce a square wave too. The signal
    is reduced to one sample per cycle by averaging non-overlapping windows of
    `window` consecutive raw samples.

    Args:
        signal: Raw signal, shape (N,) or (N, F).
        window: Number of consecutive raw samples averaged into one cycle.

    Returns:
        Downsampled signal, shape (N // window,) or (N // window, F).
        Trailing samples that do not fill a complete window are discarded.

    Raises:
        ValueError: If `window` < 1 or the signal is shorter than one window.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")

    n = signal.shape[0]
    n_cycles = n // window
    if n_cycles == 0:
        raise ValueError(
            f"signal length {n} is shorter than one window of {window}"
        )

    trimmed = signal[: n_cycles * window]
    if signal.ndim == 1:
        return trimmed.reshape(n_cycles, window).mean(axis=1)
    return trimmed.reshape(n_cycles, window, signal.shape[1]).mean(axis=1)


def downsample_only(signal: np.ndarray, downsample_window: int) -> np.ndarray:
    """
    Stage A of the split pipeline -- just average downsampling.

    Kept as a thin named wrapper so data/loader.py reads as an explicit
    three-stage pipeline (downsample -> fit stats -> standardize + smooth).

    Args:
        signal:            Raw signal, shape (N,) or (N, F).
        downsample_window: Window size for average downsampling.

    Returns:
        Downsampled signal.
    """
    return average_downsample(signal, downsample_window)


# ============================================================================
# Step 2 -- standardization  (leak-free: stats supplied by the caller)
# ============================================================================
def fit_standardizer(signals: list[np.ndarray]) -> tuple[float, float]:
    """
    Fit standardization statistics (mean, std) from one or more signals.

    The signals are flattened and pooled, so the statistics describe the whole
    training population. Call this with the TRAINING devices only -- never the
    test device -- so the test fold stays a genuine out-of-sample evaluation.

    Args:
        signals: List of arrays (any shape); each is flattened before pooling.

    Returns:
        (mean, std) as plain floats, computed over the pooled values.
    """
    pooled = np.concatenate(
        [np.asarray(s, dtype=np.float64).reshape(-1) for s in signals]
    )
    return float(pooled.mean()), float(pooled.std())


def standardize(
    signal: np.ndarray,
    mean: float | np.ndarray | None = None,
    std: float | np.ndarray | None = None,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Standardize a signal to (approximately) zero mean and unit std.

    If `mean` and `std` are given, they are applied as fixed statistics -- this
    is the leak-free path used in training/evaluation, where the stats come
    from fit_standardizer() on the training devices. If they are omitted, the
    statistics are computed from `signal` itself (per-signal standardization),
    which is convenient for standalone/exploratory use only.

    Args:
        signal: Signal to standardize, shape (C,) or (C, F).
        mean:   Fixed mean to subtract, or None to fit from `signal`.
        std:    Fixed std to divide by, or None to fit from `signal`.
        eps:    Small constant added to the std to avoid division by zero.

    Returns:
        Standardized signal, same shape as the input.
    """
    if mean is None or std is None:
        mean = signal.mean(axis=0, keepdims=True)
        std = signal.std(axis=0, keepdims=True)
    return (signal - mean) / (std + eps)


# ============================================================================
# Step 3 -- EMA window smoothing
# ============================================================================
def ema_smooth(signal: np.ndarray, span: int) -> np.ndarray:
    """
    Exponential moving average (EMA) smoothing, paper Eq. 14.

    Uses decay factor theta = 2 / (span + 1) and the adjusted EMA:

        y_t = sum_i (1 - theta)^i * x_{t-i}  /  sum_i (1 - theta)^i

    which is exactly pandas' ewm(span=..., adjust=True). Unlike a simple moving
    average, EMA places more weight on the most recent samples. Smoothing is a
    local operation, so it carries no cross-device leak.

    Args:
        signal: Standardized signal, shape (C,) or (C, F).
        span:   Sliding-window width; the paper uses 15.

    Returns:
        Smoothed signal, same shape as the input.
    """
    df = pd.DataFrame(signal)
    smoothed = df.ewm(span=span, adjust=True).mean().to_numpy()
    return smoothed.reshape(signal.shape)


def standardize_and_smooth(
    signal: np.ndarray,
    ema_span: int,
    mean: float | np.ndarray | None = None,
    std: float | np.ndarray | None = None,
    standardize_signal: bool = True,
) -> np.ndarray:
    """
    Stage C of the split pipeline -- standardize then EMA smooth.

    Run on the already-downsampled signal of a single device. Pass the fixed
    `mean` / `std` from fit_standardizer() to keep the test device leak-free.

    Args:
        signal:             Downsampled signal, shape (C,) or (C, F).
        ema_span:           Span for EMA smoothing.
        mean:               Fixed standardization mean (None -> fit locally).
        std:                Fixed standardization std  (None -> fit locally).
        standardize_signal: Whether to apply standardization (default True).

    Returns:
        Standardized + smoothed signal, same shape as the input.
    """
    x = signal
    if standardize_signal:
        x = standardize(x, mean=mean, std=std)
    x = ema_smooth(x, ema_span)
    return x


# ============================================================================
# Label construction
# ============================================================================
def build_rul_labels(
    vce: np.ndarray,
    search_fraction: float,
    drop_threshold: float,
) -> tuple[np.ndarray, int]:
    """
    Construct the normalised RUL label trajectory from a preprocessed Vce signal.

    End of life is the IGBT latch-up failure, visible as a sharp drop in the
    collector-emitter voltage. The failure cycle Nf is located as the steepest
    Vce decrease within the final `search_fraction` of the trajectory. The RUL
    label is then the straight line (paper Eq. 1):

        RUL(t) = 1 - t / Nf      for t = 0, 1, ..., Nf

    so RUL starts at 1.0 (full life) and reaches exactly 0.0 at the failure
    cycle. The caller is expected to discard any cycles after Nf (post-failure
    readings), matching the paper which cuts off data after the device fails.

    Args:
        vce:             Preprocessed Vce trajectory of one device, shape (C,).
        search_fraction: Fraction of the trajectory, measured from the end,
                         scanned for the failure drop (e.g. 0.25 -> last 25%).
        drop_threshold:  Minimum magnitude of a single Vce step, in preprocessed
                         (standardised) units, that qualifies as the failure
                         drop. If no step exceeds it, no failure is detected and
                         Nf falls back to the last cycle (no truncation).

    Returns:
        rul: np.ndarray, shape (Nf + 1,) -- RUL trajectory from 1.0 down to 0.0.
        nf:  int -- the failure cycle index (end of life).
    """
    n = vce.shape[0]
    diffs = np.diff(vce)                                  # step sizes, len n-1

    start = min(max(int((1.0 - search_fraction) * n), 0), n - 2)
    region = diffs[start:]
    steepest = float(region.min())

    if steepest <= -abs(drop_threshold):
        nf = start + int(np.argmin(region))               # failure cycle
    else:
        log.warning("no Vce drop exceeded threshold %.3f; "
                    "using the last cycle as Nf (no truncation)", drop_threshold)
        nf = n - 1

    rul = 1.0 - np.arange(nf + 1, dtype=np.float32) / float(nf)
    return rul.astype(np.float32), nf
