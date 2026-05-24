"""
training/trainer.py
===================
Training loop for IGBT RUL estimation.

Ties together the data layer, the model backbone and the physics-informed
loss, and runs fixed-epoch training with MLflow logging.

Design (decided deliberately -- see config.py "Training" section):
  * Fixed EPOCHS, no early stopping, no best-checkpoint selection.
  * The test device (leave-one-out hold-out) is evaluated every epoch for
    LOGGING ONLY. It never influences training or model selection, so the
    out-of-sample result stays honest.
  * The final model after the last epoch is the deliverable.

Baseline vs PINN is controlled entirely by config.PINN_MODE, which loss/loss_fn.py
reads -- the trainer is identical for both.

Run with:  python -m training.trainer
"""
import logging
import time

import mlflow
import torch
from torch.utils.data import TensorDataset, DataLoader

from config.config import (
    DEVICE, EPOCHS, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, SHUFFLE_TRAIN,
    SEED, DETERMINISTIC, EXPERIMENT_NAME, MLFLOW_URI,
    RUNS_DIR, REGISTRY_CSV,
    LOG_EVERY_N_EPOCHS, SAVE_FINAL_MODEL, BACKBONE, PINN_MODE,
    ALPHA, BETA, GAMMA,
    OPTIMIZER, SGD_MOMENTUM, LBFGS_MAX_ITER, LBFGS_HISTORY_SIZE,
    TEST_DEVICE, SEQ_LEN, WINDOW_STRIDE, DOWNSAMPLE_WINDOW, EMA_SPAN,
)
from utils.seed import set_seed
from models.backbone import build_backbone, count_parameters, current_arch_spec
from loss.loss_fn import compute_loss
from physics.laws import monotonicity_violations, boundary_violations
from data.loader import get_windowed_split, load_standardizer_stats
from evaluation.metrics import evaluate
from utils.run_registry import save_run, append_registry

log = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def _resolve_device(name: str) -> torch.device:
    """
    Resolve the compute device, falling back to CPU if CUDA is unavailable.

    Args:
        name: Requested device string ("cuda" or "cpu").

    Returns:
        A torch.device. If "cuda" was requested but no GPU is present, returns
        CPU and logs a warning.
    """
    if name == "cuda" and not torch.cuda.is_available():
        log.warning("CUDA requested but unavailable; falling back to CPU")
        return torch.device("cpu")
    return torch.device(name)


def _build_optimizer(model: torch.nn.Module) -> torch.optim.Optimizer:
    """
    Build the optimizer selected by config.OPTIMIZER.

    All hyperparameters are read from config.py -- this factory is the single
    place that knows how to translate the OPTIMIZER name into a constructed
    torch.optim object. To add a new optimizer, add a branch here and a docs
    entry in config.py's Optimizer section.

    Args:
        model: The network whose parameters to optimise.

    Returns:
        A torch.optim.Optimizer matching config.OPTIMIZER.

    Raises:
        ValueError: If OPTIMIZER is not recognised.
    """
    name = OPTIMIZER.lower()
    params = model.parameters()

    if name == "adam":
        return torch.optim.Adam(params, lr=LEARNING_RATE,
                                weight_decay=WEIGHT_DECAY)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=LEARNING_RATE,
                                 weight_decay=WEIGHT_DECAY)
    if name == "sgd":
        return torch.optim.SGD(params, lr=LEARNING_RATE,
                               momentum=SGD_MOMENTUM,
                               weight_decay=WEIGHT_DECAY)
    if name == "rmsprop":
        return torch.optim.RMSprop(params, lr=LEARNING_RATE,
                                   weight_decay=WEIGHT_DECAY)
    if name == "lbfgs":
        # LBFGS does not use weight_decay in the same way as first-order
        # optimisers; we pass only the inner-iteration controls.
        return torch.optim.LBFGS(params, lr=LEARNING_RATE,
                                 max_iter=LBFGS_MAX_ITER,
                                 history_size=LBFGS_HISTORY_SIZE)

    raise ValueError(
        f"unknown OPTIMIZER '{OPTIMIZER}' -- choose from "
        f"['adam', 'adamw', 'sgd', 'rmsprop', 'lbfgs']"
    )


def _is_lbfgs(optimizer: torch.optim.Optimizer) -> bool:
    """True if `optimizer` is an LBFGS instance (needs closure-based stepping)."""
    return isinstance(optimizer, torch.optim.LBFGS)


def _make_loader(
    windows: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    shuffle: bool,
) -> DataLoader:
    """
    Wrap a windowed split (inputs, targets, times, groups) in a DataLoader.

    `times` is dropped -- the loop needs only inputs, targets and groups. The
    `groups` index travels with every batch so the MDC loss can mask
    cross-device window pairs.

    Args:
        windows: 4-tuple from data.loader.make_windows / get_windowed_split.
        shuffle: Whether to shuffle. MUST be False while the MDC term is on,
                 so window order stays temporal (see config.SHUFFLE_TRAIN).

    Returns:
        A DataLoader yielding (inputs, targets, groups) batches.
    """
    inp, tgt, _times, grp = windows
    dataset = TensorDataset(inp, tgt, grp)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle)


def _run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    """
    Run one full pass over a DataLoader.

    Training pass when `optimizer` is given (forward + backward + step);
    evaluation pass otherwise (forward only, no gradients).

    Args:
        model:     The network.
        loader:    DataLoader yielding (inputs, targets, groups) batches.
        device:    Compute device.
        optimizer: Optimizer for a training pass, or None for evaluation.

    Returns:
        Dict of loss components averaged over the batches:
        'ols', 'mdc', 'bcc', 'ols_term', 'mdc_term', 'bcc_term', 'total'.
    """
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    aggregate: dict[str, float] = {}
    n_batches = 0

    # LBFGS needs a closure that re-evaluates the loss; PyTorch may call it
    # multiple times per .step(). For LBFGS we use the SAME training loop
    # structure but route the forward+backward through `optimizer.step(closure)`.
    use_closure = is_train and _is_lbfgs(optimizer)

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for inp, tgt, grp in loader:
            inp = inp.to(device)
            tgt = tgt.to(device)
            grp = grp.to(device)

            if use_closure:
                # LBFGS path: closure captures inp/tgt/grp by reference.
                # `parts_holder` lets us retrieve the loss components from the
                # last closure call for logging.
                parts_holder: dict[str, float] = {}

                def closure():
                    optimizer.zero_grad()
                    pred_c = model(inp)
                    total_c, parts_c = compute_loss(pred_c, tgt, grp)
                    total_c.backward()
                    parts_holder.update(parts_c)
                    return total_c

                optimizer.step(closure)
                parts = parts_holder
            else:
                # Standard path: forward / backward / step once per batch.
                pred = model(inp)
                total, parts = compute_loss(pred, tgt, grp)

                if is_train:
                    optimizer.zero_grad()
                    total.backward()
                    optimizer.step()

            for key, value in parts.items():
                aggregate[key] = aggregate.get(key, 0.0) + value
            n_batches += 1

    return {key: value / n_batches for key, value in aggregate.items()}


@torch.no_grad()
def _diagnose(
    model: torch.nn.Module,
    windows: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    device: torch.device,
) -> dict[str, float]:
    """
    Compute physics-violation diagnostics on a full windowed split.

    Runs the model over ALL windows at once (in their original temporal order,
    no batching) so the monotonicity check sees true consecutive pairs. This is
    the raw picture behind the squared MDC / BCC residuals -- it shows whether
    a near-zero physics loss means "constraint satisfied" or "constraint barely
    weighted".

    Args:
        model:   The network.
        windows: 4-tuple (inputs, targets, times, groups) from get_windowed_split.
        device:  Compute device.

    Returns:
        Flat dict of diagnostics, keys prefixed 'mono_' and 'bound_'.
    """
    inp, _tgt, _times, grp = windows
    model.eval()
    pred = model(inp.to(device))
    mono = monotonicity_violations(pred, grp.to(device))
    bound = boundary_violations(pred)
    return {
        "mono_n_pairs":        float(mono["n_pairs"]),
        "mono_n_violations":   float(mono["n_violations"]),
        "mono_violation_rate": mono["violation_rate"],
        "mono_max_rise":       mono["max_rise"],
        "mono_mean_rise":      mono["mean_rise"],
        "bound_n_below_0":     float(bound["n_below_0"]),
        "bound_n_above_1":     float(bound["n_above_1"]),
        "bound_min_pred":      bound["min_pred"],
        "bound_max_pred":      bound["max_pred"],
    }


# ----------------------------------------------------------------------------
# Run spec
# ----------------------------------------------------------------------------
def current_run_spec() -> dict:
    """
    Build the full run spec: architecture + data pipeline + loss + training.

    This is the complete, self-contained description of a training run. It is
    saved into the run folder (model.pt and run_info.json) and used to name the
    folder, so a saved model carries everything needed to rebuild and use it
    without depending on config.py.

    Returns:
        A dict combining the architecture spec (current_arch_spec) with the
        data-pipeline, physics-loss and training hyperparameters.
    """
    spec = dict(current_arch_spec())
    spec.update({
        "pinn_mode":         PINN_MODE,
        "test_device":       TEST_DEVICE,
        "seq_len":           SEQ_LEN,
        "window_stride":     WINDOW_STRIDE,
        "downsample_window": DOWNSAMPLE_WINDOW,
        "ema_span":          EMA_SPAN,
        "alpha":             ALPHA,
        "beta":              BETA,
        "gamma":             GAMMA,
        "epochs":            EPOCHS,
        "batch_size":        BATCH_SIZE,
        "learning_rate":     LEARNING_RATE,
        "weight_decay":      WEIGHT_DECAY,
        "seed":              SEED,
    })
    return spec


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------
def train() -> tuple[torch.nn.Module, dict[str, list[dict[str, float]]]]:
    """
    Train the model for a fixed number of epochs and return it.

    Workflow:
      1. Seed, resolve device, load the windowed leave-one-out split.
      2. Build the backbone and the Adam optimizer.
      3. For EPOCHS epochs: one training pass, then one evaluation pass on the
         test device (logged only).
      4. Save the final model to config.CKPT_DIR (if SAVE_FINAL_MODEL).

    All metrics are logged to MLflow under config.EXPERIMENT_NAME.

    Returns:
        model:   The trained network (final-epoch weights).
        history: {'train': [...], 'test': [...]} -- per-epoch loss-component
                 dicts, in epoch order.
    """
    set_seed(SEED, DETERMINISTIC)
    device = _resolve_device(DEVICE)
    log.info("device=%s | backbone=%s | PINN_MODE=%s", device, BACKBONE, PINN_MODE)

    # --- data ---------------------------------------------------------------
    train_windows, test_windows = get_windowed_split()

    # LBFGS is a full-batch optimiser: mini-batched noisy gradients break its
    # line-search assumptions. Use a single full-batch DataLoader for LBFGS so
    # each .step() sees the whole training set at once.
    is_lbfgs = OPTIMIZER.lower() == "lbfgs"
    if is_lbfgs:
        n_train_windows = train_windows[0].shape[0]
        log.info("OPTIMIZER='lbfgs' -- using full-batch training "
                 "(batch_size=%d, BATCH_SIZE in config is ignored)",
                 n_train_windows)
        train_loader = DataLoader(
            TensorDataset(train_windows[0], train_windows[1], train_windows[3]),
            batch_size=n_train_windows, shuffle=False)
    else:
        train_loader = _make_loader(train_windows, shuffle=SHUFFLE_TRAIN)

    test_loader = _make_loader(test_windows, shuffle=False)
    log.info("train windows=%d | test windows=%d",
             train_windows[0].shape[0], test_windows[0].shape[0])

    # --- model & optimizer --------------------------------------------------
    model = build_backbone().to(device)
    log.info("model parameters: %d", count_parameters(model))
    optimizer = _build_optimizer(model)
    log.info("optimizer: %s", type(optimizer).__name__)

    # --- MLflow setup -------------------------------------------------------
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    history: dict[str, list[dict[str, float]]] = {"train": [], "test": []}

    with mlflow.start_run():
        mlflow.log_params({
            "backbone": BACKBONE, "pinn_mode": PINN_MODE,
            "alpha": ALPHA, "beta": BETA, "gamma": GAMMA,
            "epochs": EPOCHS, "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
            "seq_len": SEQ_LEN, "window_stride": WINDOW_STRIDE,
            "downsample_window": DOWNSAMPLE_WINDOW, "ema_span": EMA_SPAN,
            "test_device": TEST_DEVICE, "seed": SEED,
            "model_parameters": count_parameters(model),
        })

        start = time.time()
        for epoch in range(1, EPOCHS + 1):
            train_metrics = _run_epoch(model, train_loader, device, optimizer)
            test_metrics  = _run_epoch(model, test_loader, device, None)

            # Physics-violation diagnostics on the full ordered splits.
            train_diag = _diagnose(model, train_windows, device)
            test_diag  = _diagnose(model, test_windows, device)

            history["train"].append({**train_metrics, **train_diag})
            history["test"].append({**test_metrics, **test_diag})

            mlflow.log_metrics(
                {f"train_{k}": v for k, v in train_metrics.items()}, step=epoch)
            mlflow.log_metrics(
                {f"test_{k}": v for k, v in test_metrics.items()}, step=epoch)
            mlflow.log_metrics(
                {f"train_{k}": v for k, v in train_diag.items()}, step=epoch)
            mlflow.log_metrics(
                {f"test_{k}": v for k, v in test_diag.items()}, step=epoch)

            if epoch == 1 or epoch % LOG_EVERY_N_EPOCHS == 0 or epoch == EPOCHS:
                log.info("epoch %3d/%d | train total=%.5f "
                         "(ols=%.5f mdc=%.5f bcc=%.5f) | test total=%.5f",
                         epoch, EPOCHS, train_metrics["total"],
                         train_metrics["ols"], train_metrics["mdc"],
                         train_metrics["bcc"], test_metrics["total"])
                log.info("           diag | train mono %d/%d viol (%.1f%%) "
                         "max_rise=%.4f  pred=[%.3f,%.3f] | test mono %d/%d viol",
                         int(train_diag["mono_n_violations"]),
                         int(train_diag["mono_n_pairs"]),
                         train_diag["mono_violation_rate"] * 100,
                         train_diag["mono_max_rise"],
                         train_diag["bound_min_pred"],
                         train_diag["bound_max_pred"],
                         int(test_diag["mono_n_violations"]),
                         int(test_diag["mono_n_pairs"]))

        elapsed = time.time() - start
        log.info("training finished in %.1fs", elapsed)

        # --- save the run -----------------------------------------------------
        if SAVE_FINAL_MODEL:
            # Final test-device metrics for the run folder and the registry.
            final_metrics = evaluate(model, test_windows, device)
            spec = current_run_spec()
            standardizer = load_standardizer_stats()

            folder, run_info = save_run(
                RUNS_DIR, spec, model.state_dict(),
                standardizer, final_metrics)
            append_registry(REGISTRY_CSV, run_info)

            mlflow.log_artifact(str(folder / "model.pt"))
            mlflow.log_artifact(str(folder / "run_info.json"))
            mlflow.log_metrics({f"final_{k}": v
                                for k, v in final_metrics.items()})
            log.info("run saved -> %s", folder)

    return model, history


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)-8s | %(message)s")
    train()
