"""
physics/laws.py
===============
The two physical rules for IGBT RUL estimation, as differentiable PyTorch
functions. Each returns a scalar residual -- lower means more physically
consistent. These are called by loss/loss_fn.py; they know nothing about
training, weighting, or the PINN_MODE switch.

Reproduces the constraints of
  Lu, Guo, Liu & Shi (2023), Scientific Reports 13:10167:

  * Monotonic decreasing constraint (MDC) -- paper Eqs. 8-9
  * Boundary condition constraint   (BCC) -- paper Eqs. 10-11

Unlike PDE-based PINNs, these rules are derived directly from the target RUL
function RUL(t) = 1 - t/Nf: RUL can only decrease, and normalised RUL stays
within [0, 1].
"""
import torch


def monotonic_decreasing_residual(
    predictions: torch.Tensor,
    groups: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Monotonic decreasing constraint (MDC), paper Eqs. 8-9.

    Remaining useful life must not increase as a device ages. The previous
    prediction Y_hat[i-1] should be >= the current prediction Y_hat[i]; any
    increase is penalised via ReLU(Y_hat[i] - Y_hat[i-1]):

        MSE(E_MDC) = mean( ReLU(Y_hat[i] - Y_hat[i-1])^2 )

    Device boundaries
    -----------------
    Consecutive windows in a batch may come from different devices. A drop from
    device A's last window (RUL ~ 0) to device B's first window (RUL ~ 1) is a
    legal reset, NOT a monotonicity violation. When `groups` is supplied, pairs
    that straddle a device boundary (groups[i] != groups[i-1]) are masked out,
    so only genuine within-device pairs contribute.

    This assumes `predictions` are in temporal order within each device, which
    holds when config.SHUFFLE_TRAIN is False and the data comes from
    data.loader.make_windows.

    Args:
        predictions: Model output for a batch, shape (batch,) or (batch, 1).
        groups:      Optional per-window device index, shape (batch,) or
                     (batch, 1). When given, cross-device pairs are excluded.

    Returns:
        Scalar MSE residual. Zero when no within-device prediction rises.
    """
    pred = predictions.squeeze(-1) if predictions.dim() > 1 else predictions
    if pred.numel() < 2:
        return torch.tensor(0.0, device=pred.device)

    rise = torch.relu(pred[1:] - pred[:-1])          # > 0 only when RUL rises

    if groups is not None:
        grp = groups.squeeze(-1) if groups.dim() > 1 else groups
        same_device = (grp[1:] == grp[:-1]).to(pred.dtype)   # 1 within, 0 at boundary
        rise = rise * same_device

    return (rise ** 2).mean()


def boundary_condition_residual(predictions: torch.Tensor) -> torch.Tensor:
    """
    Boundary condition constraint (BCC), paper Eqs. 10-11.

    Normalised RUL is bounded: Y_hat in [0, 1]. Error accrues only when a
    prediction leaves that range:

        MSE(E_BCC) = mean( ReLU(-Y_hat)^2 ) + mean( ReLU(Y_hat - 1)^2 )

    This is a per-prediction constraint (not a per-pair one), so it needs no
    device-boundary mask.

    Args:
        predictions: Model output, shape (batch,) or (batch, 1).

    Returns:
        Scalar MSE residual. Zero when every prediction lies within [0, 1].
    """
    pred = predictions.squeeze(-1) if predictions.dim() > 1 else predictions

    below = torch.relu(-pred)                        # > 0 when Y_hat < 0
    above = torch.relu(pred - 1.0)                   # > 0 when Y_hat > 1
    return (below ** 2).mean() + (above ** 2).mean()


# ----------------------------------------------------------------------------
# Diagnostics  (non-differentiable -- for logging / inspection only)
# ----------------------------------------------------------------------------
@torch.no_grad()
def monotonicity_violations(
    predictions: torch.Tensor,
    groups: torch.Tensor | None = None,
) -> dict[str, float]:
    """
    Count how often, and by how much, predictions violate monotonic decrease.

    The MDC residual squares each upward step, so many small violations can sum
    to a near-zero loss and look inactive in logs. This diagnostic reports the
    raw picture instead: how many pairs go up, and the size of those rises.

    Args:
        predictions: Model output, shape (batch,) or (batch, 1).
        groups:      Optional per-window device index; cross-device pairs are
                     excluded, matching monotonic_decreasing_residual.

    Returns:
        Dict with:
          'n_pairs'        -- consecutive within-device pairs considered
          'n_violations'   -- pairs where RUL increased
          'violation_rate' -- n_violations / n_pairs
          'max_rise'       -- largest single upward step (0 if none)
          'mean_rise'      -- mean upward step over violating pairs (0 if none)
    """
    pred = predictions.squeeze(-1) if predictions.dim() > 1 else predictions
    if pred.numel() < 2:
        return {"n_pairs": 0, "n_violations": 0, "violation_rate": 0.0,
                "max_rise": 0.0, "mean_rise": 0.0}

    rise = pred[1:] - pred[:-1]
    if groups is not None:
        grp = groups.squeeze(-1) if groups.dim() > 1 else groups
        rise = rise[grp[1:] == grp[:-1]]

    n_pairs = rise.numel()
    violating = rise > 0.0
    n_viol = int(violating.sum())
    rises = rise[violating]
    return {
        "n_pairs": n_pairs,
        "n_violations": n_viol,
        "violation_rate": n_viol / max(n_pairs, 1),
        "max_rise": float(rises.max()) if n_viol else 0.0,
        "mean_rise": float(rises.mean()) if n_viol else 0.0,
    }


@torch.no_grad()
def boundary_violations(predictions: torch.Tensor) -> dict[str, float]:
    """
    Count how often predictions leave the valid RUL range [0, 1].

    Like monotonicity_violations, this is the raw picture behind the squared
    BCC residual -- useful for telling "BCC is zero because the constraint is
    satisfied" apart from "BCC is zero because it is weighted away".

    Args:
        predictions: Model output, shape (batch,) or (batch, 1).

    Returns:
        Dict with:
          'n'         -- number of predictions
          'n_below_0' -- predictions below 0
          'n_above_1' -- predictions above 1
          'min_pred'  -- smallest prediction
          'max_pred'  -- largest prediction
    """
    pred = predictions.squeeze(-1) if predictions.dim() > 1 else predictions
    return {
        "n": pred.numel(),
        "n_below_0": int((pred < 0.0).sum()),
        "n_above_1": int((pred > 1.0).sum()),
        "min_pred": float(pred.min()),
        "max_pred": float(pred.max()),
    }
