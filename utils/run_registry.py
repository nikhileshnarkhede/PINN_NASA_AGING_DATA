"""
utils/run_registry.py
=====================
Per-run bookkeeping for training runs.

Each `train` run is recorded in two complementary places:

  1. A self-contained RUN FOLDER under outputs/runs/, named after the run's
     key hyperparameters, containing:
       model.pt        -- the trained weights plus the full run spec
       run_info.json   -- human-readable: spec, standardiser stats, metrics
     Because the folder name encodes the configuration, training the SAME
     configuration again overwrites that run's folder; a DIFFERENT
     configuration produces a new folder, so distinct runs never collide.

  2. A master CSV REGISTRY at outputs/runs_registry.csv, with one row per
     training run (every run appends a new row -- the CSV is a full history).
     Every hyperparameter and final metric is its own column, and the row
     records the run folder and model.pt path, so any run can be found again.

A run folder is fully self-describing: model.pt embeds the run spec, so a model
can be rebuilt and used without depending on whatever config.py currently says.
"""
import csv
import json
import logging
from datetime import datetime
from pathlib import Path

import torch

log = logging.getLogger(__name__)


def _fmt(value: object) -> str:
    """
    Render a value as a filesystem-safe token for a folder name.

    Dots become 'p' (so 0.5 -> '0p5') and spaces are stripped, keeping the
    folder name a single safe word.

    Args:
        value: Any value to embed in a folder name.

    Returns:
        A filesystem-safe string token.
    """
    return str(value).replace(".", "p").replace(" ", "")


def run_folder_name(spec: dict) -> str:
    """
    Build a descriptive, filesystem-safe folder name from a run spec.

    The name encodes the key hyperparameters so a run is identifiable at a
    glance, e.g.:
        lstm_pinn_dev2_seq25_str1_h256_l4_a0p5_b100p0_g10_e2000
    (lstm backbone, PINN mode, test device 2, seq_len 25, stride 1,
     hidden 256, 4 layers, alpha 0.5, beta 100.0, gamma 10, 2000 epochs).

    The dense-head shape and activations are NOT in the name (too long); they
    live in full inside run_info.json.

    Args:
        spec: A run spec dict (see training.trainer.current_run_spec).

    Returns:
        The folder name string.
    """
    mode = "pinn" if spec["pinn_mode"] else "baseline"
    return (
        f"{spec['backbone']}_{mode}_dev{spec['test_device']}"
        f"_seq{spec['seq_len']}_str{spec['window_stride']}"
        f"_h{spec['recurrent_hidden']}_l{spec['recurrent_layers']}"
        f"_a{_fmt(spec['alpha'])}_b{_fmt(spec['beta'])}_g{_fmt(spec['gamma'])}"
        f"_e{spec['epochs']}"
    )


def save_run(
    runs_dir: Path,
    spec: dict,
    model_state: dict,
    standardizer: dict,
    metrics: dict,
) -> tuple[Path, dict]:
    """
    Write a run's self-contained folder: model.pt and run_info.json.

    The folder is named by run_folder_name(spec). Training the same
    configuration again overwrites this folder (identical config == identical
    run); a different configuration gets its own folder.

    Args:
        runs_dir:     The outputs/runs directory.
        spec:         The full run spec (architecture + data pipeline + loss +
                      training hyperparameters).
        model_state:  model.state_dict() of the trained model.
        standardizer: The fitted standardiser stats (input_mean, input_std).
        metrics:      Final test-device metrics dict.

    Returns:
        (folder_path, run_info_dict).
    """
    runs_dir = Path(runs_dir)
    folder = runs_dir / run_folder_name(spec)
    folder.mkdir(parents=True, exist_ok=True)

    # model.pt embeds the run spec so the model is self-describing.
    model_path = folder / "model.pt"
    torch.save({"model_state": model_state, "run_spec": spec}, model_path)

    run_info = {
        "timestamp":    datetime.now().isoformat(timespec="seconds"),
        "folder":       str(folder),
        "model_path":   str(model_path),
        "spec":         spec,
        "standardizer": standardizer,
        "metrics":      metrics,
    }
    (folder / "run_info.json").write_text(json.dumps(run_info, indent=2))
    log.info("saved run folder -> %s", folder)
    return folder, run_info


# Column order for the master CSV registry.
_CSV_COLUMNS = [
    "timestamp", "folder", "model_path",
    "backbone", "pinn_mode", "test_device",
    "seq_len", "window_stride",
    "recurrent_hidden", "recurrent_layers", "recurrent_dropout",
    "dense_hidden", "dense_activation", "dense_dropout",
    "alpha", "beta", "gamma",
    "epochs", "batch_size", "learning_rate",
    "input_mean", "input_std",
    "mse", "rmse", "mae", "r2", "max_error",
]


def append_registry(csv_path: Path, run_info: dict) -> None:
    """
    Append one row for this run to the master CSV registry.

    Every training run adds a new row -- the CSV is a complete history, never
    rewritten, so re-training the same configuration still logs a fresh row
    (distinguished by its timestamp). The header is written on first use.

    List-valued parameters (dense_hidden, etc.) are stored as their string
    repr -- readable, though not numeric columns.

    Args:
        csv_path: Path to outputs/runs_registry.csv.
        run_info: The run_info dict returned by save_run.
    """
    csv_path = Path(csv_path)
    spec = run_info["spec"]
    std = run_info["standardizer"]
    metrics = run_info["metrics"]

    row = {
        "timestamp":         run_info["timestamp"],
        "folder":            run_info["folder"],
        "model_path":        run_info["model_path"],
        "backbone":          spec["backbone"],
        "pinn_mode":         spec["pinn_mode"],
        "test_device":       spec["test_device"],
        "seq_len":           spec["seq_len"],
        "window_stride":     spec["window_stride"],
        "recurrent_hidden":  spec["recurrent_hidden"],
        "recurrent_layers":  spec["recurrent_layers"],
        "recurrent_dropout": spec["recurrent_dropout"],
        "dense_hidden":      str(spec["dense_hidden"]),
        "dense_activation":  str(spec["dense_activation"]),
        "dense_dropout":     str(spec["dense_dropout"]),
        "alpha":             spec["alpha"],
        "beta":              spec["beta"],
        "gamma":             spec["gamma"],
        "epochs":            spec["epochs"],
        "batch_size":        spec["batch_size"],
        "learning_rate":     spec["learning_rate"],
        "input_mean":        std.get("input_mean"),
        "input_std":         std.get("input_std"),
        "mse":               metrics.get("mse"),
        "rmse":              metrics.get("rmse"),
        "mae":               metrics.get("mae"),
        "r2":                metrics.get("r2"),
        "max_error":         metrics.get("max_error"),
    }

    write_header = not csv_path.exists()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    log.info("appended run to registry -> %s", csv_path)
