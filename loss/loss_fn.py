"""
loss/loss_fn.py
===============
Assembles the total training loss from the data term and the two physics
constraints, reproducing the physics-informed loss of
  Lu, Guo, Liu & Shi (2023), Scientific Reports 13:10167, Eq. 6:

    E_PINN = (1 - alpha) * OLS  +  alpha * gamma * MDC  +  beta * BCC

  OLS -- ordinary least squares: the data fit error  (paper Eq. 7)
  MDC -- monotonic decreasing constraint              (paper Eq. 9)
  BCC -- boundary condition constraint                (paper Eq. 11)

PINN_MODE switch
----------------
config.PINN_MODE is the master switch between baseline and physics-informed
training. The collapse to baseline happens HERE, not in config:

    _alpha = ALPHA if PINN_MODE else 0.0
    _beta  = BETA  if PINN_MODE else 0.0

When PINN_MODE is False, _alpha = _beta = 0, so the total loss reduces to pure
OLS -- the plain baseline RNN/LSTM. ALPHA / BETA in config.py keep their values
either way; they are simply ignored when PINN_MODE is False. To flip between a
baseline run and a PINN run, change only config.PINN_MODE.

The raw OLS / MDC / BCC residuals are always computed and returned for logging,
regardless of mode, so training curves show every component.
"""
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from config.config import PINN_MODE, ALPHA, BETA, GAMMA, OLS_LOSS, HUBER_DELTA
from physics.laws import (
    monotonic_decreasing_residual,
    boundary_condition_residual,
)

log = logging.getLogger(__name__)

# Effective weights -- this is what loss_fn.py actually reads.
# PINN_MODE False -> physics terms are switched off (pure baseline OLS).
_alpha: float = ALPHA if PINN_MODE else 0.0
_beta:  float = BETA  if PINN_MODE else 0.0
_gamma: float = GAMMA          # only scales MDC, which is already 0 when _alpha = 0


# ----------------------------------------------------------------------------
# Data-term (OLS) loss factory  -- config.OLS_LOSS selects which one
# ----------------------------------------------------------------------------
class _LogCoshLoss(nn.Module):
    """
    Numerically stable log(cosh(error)) regression loss.

    log(cosh(x)) ~ x^2 / 2 near zero (like MSE) and ~ |x| - log(2) for large x
    (like MAE). Implemented as `x + softplus(-2x) - log(2)` to avoid overflow.
    """

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        x = pred - target
        return (x + F.softplus(-2.0 * x)
                - torch.log(torch.tensor(2.0, device=x.device))).mean()


def _build_ols_criterion() -> nn.Module:
    """
    Build the data-term loss selected by config.OLS_LOSS.

    Replaces the OLS term ONLY. The physics terms (MDC, BCC) are defined by
    the paper as squared-ReLU residuals and are not affected by this choice.

    Returns:
        A torch.nn.Module mapping (pred, target) -> scalar loss.

    Raises:
        ValueError: If OLS_LOSS is not a recognised option.
    """
    name = OLS_LOSS.lower()
    if name == "mse":
        return nn.MSELoss()
    if name == "mae":
        return nn.L1Loss()
    if name == "huber":
        return nn.HuberLoss(delta=HUBER_DELTA)
    if name == "smooth_l1":
        return nn.SmoothL1Loss()
    if name == "log_cosh":
        return _LogCoshLoss()
    raise ValueError(
        f"unknown OLS_LOSS '{OLS_LOSS}' -- choose from "
        f"['mse', 'mae', 'huber', 'smooth_l1', 'log_cosh']"
    )


_ols_criterion = _build_ols_criterion()

log.info("loss configured | PINN_MODE=%s  alpha=%.3g  beta=%.3g  gamma=%.3g  "
         "OLS_LOSS=%s", PINN_MODE, _alpha, _beta, _gamma, OLS_LOSS)


def compute_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    groups: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Compute the total loss for one batch of RUL predictions.

    Combines the three terms with the effective weights:

        total = (1 - _alpha) * OLS  +  _alpha * _gamma * MDC  +  _beta * BCC

    In baseline mode (PINN_MODE False) _alpha = _beta = 0, so total = OLS.

    Args:
        predictions: Model output, shape (batch,) or (batch, 1).
        targets:     Ground-truth normalised RUL, shape (batch,) or (batch, 1).
        groups:      Optional per-window device index, shape (batch,). Passed
                     to the MDC term so it skips cross-device pairs (see
                     physics.laws.monotonic_decreasing_residual). Required for
                     a correct MDC when a batch spans multiple devices.

    Returns:
        total: Scalar tensor -- the value to call .backward() on.
        parts: Dict of float components for logging:
               'ols', 'mdc', 'bcc'            -- raw residuals (unweighted)
               'ols_term', 'mdc_term', 'bcc_term' -- weighted contributions
               'total'                        -- the scalar above
    """
    pred = predictions.squeeze(-1) if predictions.dim() > 1 else predictions
    tgt  = targets.squeeze(-1) if targets.dim() > 1 else targets

    # Raw residuals -- always computed so logging shows every component.
    ols = _ols_criterion(pred, tgt)                          # paper Eq. 7 (form chosen by OLS_LOSS)
    mdc = monotonic_decreasing_residual(pred, groups)        # paper Eq. 9
    bcc = boundary_condition_residual(pred)                  # paper Eq. 11

    # Weighted total -- paper Eq. 6.
    ols_term = (1.0 - _alpha) * ols
    mdc_term = _alpha * _gamma * mdc
    bcc_term = _beta * bcc
    total = ols_term + mdc_term + bcc_term

    parts = {
        "ols": ols.item(),
        "mdc": mdc.item(),
        "bcc": bcc.item(),
        "ols_term": ols_term.item(),
        "mdc_term": mdc_term.item(),
        "bcc_term": bcc_term.item(),
        "total": total.item(),
    }
    return total, parts
