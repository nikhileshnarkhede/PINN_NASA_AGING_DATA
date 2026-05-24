"""
evaluation/metrics.py
=====================
Regression-quality metrics for IGBT RUL estimation.

Provides the two metrics reported in
  Lu, Guo, Liu & Shi (2023), Scientific Reports 13:10167 -- MSE and R^2
  (coefficient of determination, paper Eq. 13) -- plus RMSE, MAE and the
  maximum absolute error.

All metric functions take prediction / target tensors of any matching shape
(they are flattened internally) and return plain Python floats. The convenience
function evaluate() runs a model over a windowed split and returns every metric
in one dict.

Note -- MAPE is deliberately excluded. The RUL target reaches exactly 0.0 at
end of life, so a percentage error divides by zero and explodes near the most
important region. It is not a valid metric for this target.
"""
import logging

import torch
import torch.nn as nn

log = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Individual metrics
# ----------------------------------------------------------------------------
def mse(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Mean squared error -- the paper's primary metric (Eq. 5 / 7).

    Args:
        predictions: Predicted RUL, any shape.
        targets:     Ground-truth RUL, same number of elements.

    Returns:
        Mean of squared differences, as a float.
    """
    pred = predictions.reshape(-1)
    tgt = targets.reshape(-1)
    return float(torch.mean((tgt - pred) ** 2))


def rmse(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Root mean squared error -- MSE in the original RUL units (0 to 1).

    Interpretable as a typical prediction error: an RMSE of 0.1 means the model
    is off by roughly 10% of total device life on average.

    Args:
        predictions: Predicted RUL, any shape.
        targets:     Ground-truth RUL, same number of elements.

    Returns:
        Square root of the MSE, as a float.
    """
    return float(mse(predictions, targets) ** 0.5)


def mae(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Mean absolute error -- average absolute deviation, robust to outliers.

    Args:
        predictions: Predicted RUL, any shape.
        targets:     Ground-truth RUL, same number of elements.

    Returns:
        Mean of absolute differences, as a float.
    """
    pred = predictions.reshape(-1)
    tgt = targets.reshape(-1)
    return float(torch.mean(torch.abs(tgt - pred)))


def max_error(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Maximum absolute error -- the single worst prediction.

    Worth reporting for a safety-relevant quantity like RUL: it answers
    "what is the worst-case miss?", which a mean metric hides.

    Args:
        predictions: Predicted RUL, any shape.
        targets:     Ground-truth RUL, same number of elements.

    Returns:
        Largest absolute difference, as a float.
    """
    pred = predictions.reshape(-1)
    tgt = targets.reshape(-1)
    return float(torch.max(torch.abs(tgt - pred)))


def r2_score(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Coefficient of determination R^2 -- the paper's second metric (Eq. 13):

        R^2 = 1 - sum((Y - Y_hat)^2) / sum((Y - Y_mean)^2)

    1.0 is a perfect fit; 0.0 means the model is no better than always
    predicting the mean; negative means it is worse than that.

    Args:
        predictions: Predicted RUL, any shape.
        targets:     Ground-truth RUL, same number of elements.

    Returns:
        R^2 as a float. Returns nan if the targets have zero variance
        (R^2 is undefined in that degenerate case).
    """
    pred = predictions.reshape(-1)
    tgt = targets.reshape(-1)
    ss_res = torch.sum((tgt - pred) ** 2)
    ss_tot = torch.sum((tgt - torch.mean(tgt)) ** 2)
    if ss_tot == 0:
        log.warning("R^2 undefined: targets have zero variance")
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


# ----------------------------------------------------------------------------
# Convenience: evaluate a model on a windowed split
# ----------------------------------------------------------------------------
def evaluate(
    model: nn.Module,
    windows: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    device: torch.device,
) -> dict[str, float]:
    """
    Run a model over a windowed split and return all regression metrics.

    Predictions are pooled over every window in the split (for the leave-one-out
    setup this is the single test device). The function is structured so a
    per-device breakdown can be added later without changing the metric code.

    Args:
        model:   Trained network.
        windows: 4-tuple (inputs, targets, times, groups) from
                 data.loader.get_windowed_split.
        device:  Compute device.

    Returns:
        Dict with keys 'mse', 'rmse', 'mae', 'r2', 'max_error'.
    """
    inputs, targets, _times, _groups = windows

    model.eval()
    with torch.no_grad():
        predictions = model(inputs.to(device)).cpu()
    targets = targets.cpu()

    return {
        "mse":       mse(predictions, targets),
        "rmse":      rmse(predictions, targets),
        "mae":       mae(predictions, targets),
        "r2":        r2_score(predictions, targets),
        "max_error": max_error(predictions, targets),
    }
