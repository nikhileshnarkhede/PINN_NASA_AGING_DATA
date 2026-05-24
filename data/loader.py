"""
data/loader.py
==============
Loads the NASA PCoE IGBT accelerated-aging dataset and builds the
leave-one-device-out train / test split used for out-of-sample evaluation.

Public functions
----------------
  load_nasa_igbt()       -- raw extraction from the .mat files (no preprocessing)
  train_val_test_split() -- preprocessing + RUL labels + leave-one-out split

Leak-free standardization
-------------------------
All four devices are the same IGBT model, so the trained model is meant to be
device-agnostic: at deployment it sees only a window of Vce values and returns
RUL, with no device identity. To test that honestly the held-out device must
not influence its own normalisation. train_val_test_split() therefore:

  Stage A -- downsamples every device           (no cross-device statistics)
  Stage B -- fits standardization mu / sigma on the TRAIN devices only,
             and saves them to outputs/standardizer_stats.json
  Stage C -- standardizes + smooths every device with those fixed stats,
             builds RUL labels, and truncates each device at its failure cycle

The saved standardizer_stats.json is a deployment artifact: the inference path
must load it to scale incoming signals exactly as during training.

All settings are read from config/config.py. Run a quick check + plot with:

    python -m data.loader
"""
import glob
import json
import logging
from pathlib import Path

import numpy as np
import scipy.io
import torch

from config.config import (
    DATA_DIR, DEVICE_FILE_GLOB, DEVICES, TEST_DEVICE,
    RAW_FEATURES, VCE_FIELD, INPUT_FEATURES,
    DROP_NAN_ROWS, DOWNSAMPLE_WINDOW, STANDARDIZE, EMA_SPAN,
    FAILURE_SEARCH_FRACTION, FAILURE_DROP_THRESHOLD,
    PAD_VALUE, TIME_PAD_SENTINEL, OUTPUT_DIR, SEQ_LEN, WINDOW_STRIDE,
)
from data.features import (
    downsample_only, standardize_and_smooth, fit_standardizer, build_rul_labels,
)

log = logging.getLogger(__name__)

# Filename of the saved standardizer statistics (a deployment artifact).
STANDARDIZER_STATS_FILE = "standardizer_stats.json"


# ----------------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------------
def _find_device_file(data_dir: Path, device: int) -> Path:
    """
    Locate the .mat file for a device, tolerating inconsistent filename spacing.

    Args:
        data_dir: Directory holding the .mat files.
        device:   Device number.

    Returns:
        Path to the matched .mat file.

    Raises:
        FileNotFoundError: If no file matches the device glob pattern.
    """
    pattern = str(Path(data_dir) / DEVICE_FILE_GLOB.format(device=device))
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No .mat file found for device {device} (pattern: {pattern})"
        )
    if len(matches) > 1:
        log.warning("Multiple files for device %d; using %s",
                    device, matches[0])
    return Path(matches[0])


def _extract_device(path: Path) -> np.ndarray:
    """
    Extract the RAW_FEATURES matrix from one device .mat file.

    Mirrors the tested notebook logic: open the `measurement` struct, walk
    `steadyState`, and read the scalar `timeDomain` fields from each entry.

    Args:
        path: Path to the device .mat file.

    Returns:
        Array of shape (N, len(RAW_FEATURES)). Rows with a nan Vce value are
        dropped when config.DROP_NAN_ROWS is True.
    """
    mat = scipy.io.loadmat(str(path), struct_as_record=False, squeeze_me=True)
    steady_states = mat["measurement"].steadyState

    rows = [[float(getattr(ss.timeDomain, f)) for f in RAW_FEATURES]
            for ss in steady_states]
    arr = np.asarray(rows, dtype=np.float64)

    if DROP_NAN_ROWS:
        vce_col = RAW_FEATURES.index(VCE_FIELD)
        keep = ~np.isnan(arr[:, vce_col])
        dropped = int((~keep).sum())
        if dropped:
            log.info("%s: dropped %d nan placeholder row(s)", path.name, dropped)
        arr = arr[keep]
    return arr


def _pad_stack(arrays: list[np.ndarray], pad_value: float) -> np.ndarray:
    """
    Right-pad a list of (L_i, ...) arrays and stack into one (n, L_max, ...) array.

    Args:
        arrays:    List of arrays sharing every dimension except the first.
        pad_value: Value used to fill padded positions.

    Returns:
        Stacked float32 array of shape (len(arrays), L_max, ...).
    """
    max_len = max(a.shape[0] for a in arrays)
    tail = arrays[0].shape[1:]
    out = np.full((len(arrays), max_len, *tail), pad_value, dtype=np.float32)
    for i, a in enumerate(arrays):
        out[i, : a.shape[0]] = a
    return out


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------
def load_nasa_igbt(
    data_dir: Path = DATA_DIR,
    devices: list[int] = DEVICES,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Raw extraction from the NASA IGBT .mat files (no preprocessing applied).

    Devices have different raw lengths, so the per-device sequences are
    right-padded into one stacked tensor. Padded positions hold config.PAD_VALUE
    in `inputs` / `targets` and config.TIME_PAD_SENTINEL in `times`; a device's
    true length is recoverable via (times >= 0).

    Args:
        data_dir: Directory holding the .mat files.
        devices:  Device numbers to load (output order follows this list).

    Returns:
        inputs:  torch.Tensor, shape (n_devices, seq_len, n_features) --
                 the INPUT_FEATURES signals.
        targets: torch.Tensor, shape (n_devices, seq_len) --
                 raw Vce (collector-emitter voltage) values.
        times:   torch.Tensor, shape (n_devices, seq_len) --
                 raw sample index (TIME_PAD_SENTINEL on padded positions).
    """
    in_idx  = [RAW_FEATURES.index(f) for f in INPUT_FEATURES]
    vce_idx = RAW_FEATURES.index(VCE_FIELD)

    inp_list, tgt_list, time_list = [], [], []
    for dev in devices:
        path = _find_device_file(data_dir, dev)
        arr = _extract_device(path)                          # (N, 8)
        n = arr.shape[0]
        inp_list.append(arr[:, in_idx].astype(np.float32))   # (N, F)
        tgt_list.append(arr[:, vce_idx].astype(np.float32))  # (N,)
        time_list.append(np.arange(n, dtype=np.float32))     # (N,)
        log.info("device %d: %d raw samples", dev, n)

    inputs  = _pad_stack(inp_list, PAD_VALUE)
    targets = _pad_stack([t[:, None] for t in tgt_list], PAD_VALUE)[..., 0]
    times   = _pad_stack([t[:, None] for t in time_list], TIME_PAD_SENTINEL)[..., 0]

    return (torch.from_numpy(inputs),
            torch.from_numpy(targets),
            torch.from_numpy(times))


def train_val_test_split(
    data_dir: Path = DATA_DIR,
    devices: list[int] = DEVICES,
    test_device: int = TEST_DEVICE,
) -> tuple[
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]:
    """
    Build the leave-one-device-out train / test split with leak-free
    preprocessing and RUL labels.

    Three stages (see the module docstring):
      A. Downsample every device -- per-device, no cross-device statistics.
      B. Fit standardization mu / sigma on the TRAIN devices only and persist
         them to outputs/standardizer_stats.json.
      C. Standardize (with the fixed train stats) + EMA-smooth every device,
         build the RUL label trajectory, and truncate each device at its
         failure cycle Nf so the post-failure Vce collapse is discarded.

    With the default config this gives train = devices 2, 3, 5 and
    test = device 4.

    Note on tensor meanings -- `targets` is the RUL label here, NOT Vce.
    The Vce signal the model consumes is carried in `inputs`. The test device
    is normalised with the train devices' statistics, so it never influences
    its own scaling -- a genuine out-of-sample evaluation.

    Args:
        data_dir:    Directory holding the .mat files.
        devices:     Device numbers to load.
        test_device: Device held out for testing.

    Returns:
        ((train_inputs, train_targets, train_times),
         (test_inputs,  test_targets,  test_times))

        Each tensor is right-padded across its group:
          inputs  -- (n_devices, seq_len, n_features), preprocessed Vce
          targets -- (n_devices, seq_len), RUL label (1.0 down to 0.0)
          times   -- (n_devices, seq_len), cycle index
                     (TIME_PAD_SENTINEL on padded positions)
    """
    inputs, targets, times = load_nasa_igbt(data_dir, devices)

    # --- Stage A: downsample every device (no cross-device statistics) -------
    ds_inp: dict[int, np.ndarray] = {}
    ds_vce: dict[int, np.ndarray] = {}
    for d, dev in enumerate(devices):
        valid = times[d].numpy() >= 0
        ds_inp[dev] = downsample_only(inputs[d].numpy()[valid], DOWNSAMPLE_WINDOW)
        ds_vce[dev] = downsample_only(targets[d].numpy()[valid], DOWNSAMPLE_WINDOW)

    # --- Stage B: fit standardizer on TRAIN devices only (leak-free) ---------
    train_dev_list = [dev for dev in devices if dev != test_device]
    in_mean,  in_std  = fit_standardizer([ds_inp[dev] for dev in train_dev_list])
    vce_mean, vce_std = fit_standardizer([ds_vce[dev] for dev in train_dev_list])
    log.info("standardizer fitted on train devices %s | "
             "input mu=%.4f sigma=%.4f", train_dev_list, in_mean, in_std)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stats = {
        "train_devices": train_dev_list,
        "test_device": test_device,
        "input_mean": in_mean, "input_std": in_std,
        "vce_mean": vce_mean, "vce_std": vce_std,
    }
    stats_path = OUTPUT_DIR / STANDARDIZER_STATS_FILE
    with open(stats_path, "w") as fh:
        json.dump(stats, fh, indent=2)
    log.info("saved standardizer stats -> %s", stats_path)

    # --- Stage C: standardize + smooth + RUL label, per device --------------
    train: tuple[list, list, list] = ([], [], [])
    test:  tuple[list, list, list] = ([], [], [])
    for dev in devices:
        proc_inp = standardize_and_smooth(
            ds_inp[dev], EMA_SPAN, in_mean, in_std, STANDARDIZE)      # (C, F)
        proc_vce = standardize_and_smooth(
            ds_vce[dev], EMA_SPAN, vce_mean, vce_std, STANDARDIZE)    # (C,)

        rul, nf = build_rul_labels(
            proc_vce, FAILURE_SEARCH_FRACTION, FAILURE_DROP_THRESHOLD)
        proc_inp  = proc_inp[: nf + 1]                               # (Nf+1, F)
        proc_time = np.arange(nf + 1, dtype=np.float32)              # (Nf+1,)

        bucket = test if dev == test_device else train
        bucket[0].append(proc_inp.astype(np.float32))                # inputs = Vce
        bucket[1].append(rul.astype(np.float32))                     # targets = RUL
        bucket[2].append(proc_time)                                  # times = cycle
        log.info("device %d -> %-5s | Nf=%d  (%d cycles, %d discarded post-failure)",
                 dev, "test" if dev == test_device else "train",
                 nf, proc_vce.shape[0], proc_vce.shape[0] - (nf + 1))

    def _bundle(group: tuple[list, list, list]):
        i = torch.from_numpy(_pad_stack(group[0], PAD_VALUE))
        t = torch.from_numpy(
            _pad_stack([x[:, None] for x in group[1]], PAD_VALUE)[..., 0])
        m = torch.from_numpy(
            _pad_stack([x[:, None] for x in group[2]], TIME_PAD_SENTINEL)[..., 0])
        return i, t, m

    return _bundle(train), _bundle(test)


def load_standardizer_stats() -> dict:
    """
    Read the standardiser statistics saved by the most recent split.

    train_val_test_split() writes outputs/standardizer_stats.json every time it
    runs. This accessor lets the trainer embed those stats (input_mean /
    input_std) into a run folder, so a saved model carries the exact scaling
    its inputs were trained with.

    Returns:
        The stats dict (train_devices, test_device, input_mean, input_std,
        vce_mean, vce_std), or an empty dict if the file does not exist yet.
    """
    stats_path = OUTPUT_DIR / STANDARDIZER_STATS_FILE
    if not stats_path.exists():
        log.warning("standardizer stats file not found: %s", stats_path)
        return {}
    with open(stats_path) as fh:
        return json.load(fh)


# ----------------------------------------------------------------------------
# Windowing  (many-to-one sequences for the RNN)
# ----------------------------------------------------------------------------
def make_windows(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    times: torch.Tensor,
    seq_len: int,
    stride: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Slice padded per-device trajectories into overlapping many-to-one windows.

    The paper's model is many-to-one with time step s = seq_len: it consumes
    `seq_len` consecutive Vce values and predicts the RUL at the window's last
    cycle. Successive windows advance by `stride` cycles, so a device with C
    cycles yields (C - seq_len) // stride + 1 windows. Windows from every
    device are concatenated into one flat batch, in device order, with cycle
    order preserved within each device.

    Device boundaries and the MDC loss
    ----------------------------------
    The monotonic-decreasing constraint compares each prediction with the
    previous one. Across a device boundary that comparison is meaningless --
    e.g. device 2's last window (RUL ~ 0) followed by device 3's first window
    (RUL ~ 1) is a legal jump, not a violation. To let the loss skip such
    pairs, this function also returns a per-window `groups` index. The MDC loss
    must only penalise consecutive windows i-1, i when groups[i-1] == groups[i].

    Args:
        inputs:  Padded device inputs,  shape (n_devices, max_cycles, n_features).
        targets: Padded device RUL,     shape (n_devices, max_cycles).
        times:   Padded device cycles,  shape (n_devices, max_cycles)
                 (TIME_PAD_SENTINEL marks padding).
        seq_len: Window length (paper: s = 10).
        stride:  Cycles advanced between consecutive windows (1 = max overlap).

    Returns:
        win_inputs:  torch.Tensor, shape (n_windows, seq_len, n_features) --
                     the Vce window fed to the model.
        win_targets: torch.Tensor, shape (n_windows,) --
                     RUL at each window's last cycle (the supervised label).
        win_times:   torch.Tensor, shape (n_windows,) --
                     cycle index of each window's last cycle.
        win_groups:  torch.Tensor (long), shape (n_windows,) --
                     device index each window came from; used to mask the MDC
                     loss at device boundaries.
    """
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")

    win_inp: list[torch.Tensor] = []
    win_tgt: list[torch.Tensor] = []
    win_tm:  list[torch.Tensor] = []
    win_grp: list[int] = []

    for d in range(inputs.shape[0]):
        valid = times[d] >= 0                       # strip right-padding
        dev_inp = inputs[d][valid]                  # (C, F)
        dev_tgt = targets[d][valid]                 # (C,)
        dev_tm  = times[d][valid]                   # (C,)
        cycles = dev_inp.shape[0]

        if cycles < seq_len:
            log.warning("device index %d has %d cycles < seq_len %d; skipped",
                        d, cycles, seq_len)
            continue

        for s in range(0, cycles - seq_len + 1, stride):
            win_inp.append(dev_inp[s: s + seq_len])             # (seq_len, F)
            win_tgt.append(dev_tgt[s + seq_len - 1])            # scalar RUL
            win_tm.append(dev_tm[s + seq_len - 1])              # last cycle idx
            win_grp.append(d)                                   # device group

    if not win_inp:
        raise ValueError(
            f"no windows produced -- every device shorter than seq_len={seq_len}"
        )

    return (torch.stack(win_inp),
            torch.stack(win_tgt),
            torch.stack(win_tm),
            torch.tensor(win_grp, dtype=torch.long))


def get_windowed_split(
    data_dir: Path = DATA_DIR,
    devices: list[int] = DEVICES,
    test_device: int = TEST_DEVICE,
    seq_len: int | None = None,
    stride: int | None = None,
) -> tuple[
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
]:
    """
    One-call entry point: leave-one-out split -> windowed many-to-one tensors.

    Runs train_val_test_split() (preprocessing + RUL labels + LOO split) and
    then make_windows() on each side. This is the function the training script
    should call to obtain model-ready data.

    Args:
        data_dir:    Directory holding the .mat files.
        devices:     Device numbers to load.
        test_device: Device held out for testing.
        seq_len:     Window length; defaults to config.SEQ_LEN.
        stride:      Cycles advanced between windows; defaults to
                     config.WINDOW_STRIDE.

    Returns:
        (train_windows, test_windows), where each is the 4-tuple returned by
        make_windows(): (win_inputs, win_targets, win_times, win_groups).
    """
    seq_len = SEQ_LEN if seq_len is None else seq_len
    stride  = WINDOW_STRIDE if stride is None else stride

    (tr_in, tr_tg, tr_tm), (te_in, te_tg, te_tm) = train_val_test_split(
        data_dir, devices, test_device)

    train_windows = make_windows(tr_in, tr_tg, tr_tm, seq_len, stride)
    test_windows  = make_windows(te_in, te_tg, te_tm, seq_len, stride)

    log.info("windowed (seq_len=%d, stride=%d) | train: %d windows  test: %d windows",
             seq_len, stride, train_windows[0].shape[0], test_windows[0].shape[0])
    return train_windows, test_windows


if __name__ == "__main__":
    # Shape check + a per-device plot of the model input (Vce) and the model
    # target (RUL):  python -m data.loader
    import math
    import matplotlib.pyplot as plt

    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)-8s | %(message)s")

    raw_inputs, raw_targets, raw_times = load_nasa_igbt()
    log.info("RAW   | inputs %s  targets %s  times %s",
             tuple(raw_inputs.shape), tuple(raw_targets.shape),
             tuple(raw_times.shape))

    (tr_in, tr_tg, tr_tm), (te_in, te_tg, te_tm) = train_val_test_split()
    log.info("TRAIN | inputs %s  targets %s  times %s",
             tuple(tr_in.shape), tuple(tr_tg.shape), tuple(tr_tm.shape))
    log.info("TEST  | inputs %s  targets %s  times %s",
             tuple(te_in.shape), tuple(te_tg.shape), tuple(te_tm.shape))

    # Collect (cycle, Vce input, RUL target) per device, padding stripped.
    per_device: dict[int, tuple] = {}
    train_devices = [d for d in DEVICES if d != TEST_DEVICE]
    for idx, dev in enumerate(train_devices):
        keep = tr_tm[idx].numpy() >= 0
        per_device[dev] = (tr_tm[idx].numpy()[keep],
                           tr_in[idx, :, 0].numpy()[keep],
                           tr_tg[idx].numpy()[keep])
    keep = te_tm[0].numpy() >= 0
    per_device[TEST_DEVICE] = (te_tm[0].numpy()[keep],
                               te_in[0, :, 0].numpy()[keep],
                               te_tg[0].numpy()[keep])

    # One panel per device: Vce (blue, left axis) and RUL (red, right axis).
    devices_sorted = sorted(per_device)
    panels = "ABCDEFGH"
    ncols = 2
    nrows = math.ceil(len(devices_sorted) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 3.5 * nrows),
                             squeeze=False)
    for ax, panel, dev in zip(axes.flat, panels, devices_sorted):
        cycle, vce, rul = per_device[dev]
        l1 = ax.plot(cycle, vce, color="tab:blue", linewidth=1.0,
                     label="Vce (input)")
        ax.set_xlabel("Cycle")
        ax.set_ylabel("Collector-emitter voltage (standardized)")
        ax.grid(True, alpha=0.3)

        ax2 = ax.twinx()
        l2 = ax2.plot(cycle, rul, color="tab:red", linewidth=1.2,
                      label="RUL (target)")
        ax2.set_ylabel("RUL")
        ax2.set_ylim(-0.05, 1.05)

        role = "test" if dev == TEST_DEVICE else "train"
        ax.set_title(f"({panel}) Device {dev}  [{role}]")
        lines = l1 + l2
        ax.legend(lines, [ln.get_label() for ln in lines],
                  loc="center right", fontsize=8)
    for ax in list(axes.flat)[len(devices_sorted):]:
        ax.set_visible(False)
    fig.suptitle("Model input (Vce) and model target (RUL) - all devices",
                 fontsize=12)
    fig.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "vce_and_rul_all_devices.png"
    fig.savefig(out_path, dpi=150)
    log.info("saved plot -> %s", out_path)
    plt.show()
