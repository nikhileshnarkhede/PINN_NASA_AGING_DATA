"""
evaluation/plot_rul.py
======================
Plot actual vs predicted RUL against cycle, for every saved checkpoint.

Draws one figure for the test device: the actual RUL trajectory (the
1 - t/Nf ground truth) plus one predicted-RUL line per checkpoint found in
config.CKPT_DIR. This is the most informative diagnostic for the project --
it shows directly how each model tracks (or departs from) the true RUL curve.

Each checkpoint is rebuilt from its own saved architecture spec, so models of
different architectures plot correctly side by side. The figure is written to
outputs/ in every format listed in config.PLOT_FORMATS.

Standalone:  python -m evaluation.plot_rul
Via the CLI: python -m main plot
"""
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                       # headless-safe backend
import matplotlib.pyplot as plt
import torch

from config.config import (
    RUNS_DIR, OUTPUT_DIR, TEST_DEVICE, PLOT_FORMATS, AUTO_OPEN_PLOTS,
)
from models.backbone import build_backbone_from_spec
from data.loader import get_windowed_split
from utils.open_file import open_in_viewer

log = logging.getLogger(__name__)

import warnings
# Suppress all FutureWarnings (common in data science libraries like Pandas)
warnings.simplefilter(action='ignore', category=FutureWarning)
from warnings import filterwarnings
filterwarnings("ignore")


def _load_run_model(model_path: Path, device: torch.device) -> torch.nn.Module:
    """
    Rebuild a model from a run folder's model.pt.

    model.pt embeds the run spec, so the model is rebuilt exactly as trained --
    independent of config.py.

    Args:
        model_path: Path to a run folder's model.pt.
        device:     Device to place the model on.

    Returns:
        The model with loaded weights, on `device`, in eval mode.
    """
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    model = build_backbone_from_spec(ckpt["run_spec"])
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model


def plot_rul() -> list[Path]:
    """
    Plot actual vs predicted RUL for every checkpoint in config.CKPT_DIR.

    Builds one figure for the test device, with the actual RUL and each
    model's prediction drawn against cycle, and writes it to outputs/ in each
    format in config.PLOT_FORMATS.

    A checkpoint that cannot be loaded (e.g. a corrupt file) is skipped with a
    warning rather than aborting the whole plot.

    Returns:
        List of written figure paths (empty if there were no checkpoints).
    """
    device = torch.device("cpu")             # plotting is light; CPU is fine

    # Test device windows: predictions are aligned to each window's last cycle.
    _train_windows, test_windows = get_windowed_split()
    _inputs, targets, times, _groups = test_windows
    cycles = times.numpy()
    actual = targets.numpy()

    checkpoints = sorted(Path(RUNS_DIR).glob("*/model.pt"))
    if not checkpoints:
        log.error("no runs found in %s -- train a model first "
                  "(python -m main train)", RUNS_DIR)
        return []

    fig, ax = plt.subplots(figsize=(9.0, 5.5))
    ax.plot(cycles, actual, color="black", linewidth=2.2,
            label="Actual RUL", zorder=10)

    plotted = 0
    for model_path in checkpoints:
        run_name = model_path.parent.name
        try:
            model = _load_run_model(model_path, device)
            with torch.no_grad():
                predictions = model(test_windows[0].to(device)).cpu().numpy()
        except Exception as exc:             # corrupt file, bad spec, etc.
            log.warning("skipping %s: %s", run_name, exc)
            continue
        ax.plot(cycles, predictions, linewidth=1.4, alpha=0.85,
                label=run_name)
        plotted += 1

    if plotted == 0:
        log.error("no checkpoint could be loaded; nothing to plot")
        plt.close(fig)
        return []

    ax.set_xlabel("Cycle")
    ax.set_ylabel("RUL")
    ax.set_title(f"Actual vs predicted RUL  --  test device {TEST_DEVICE}")
    ax.set_ylim(-0.05, 1.1)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for fmt in PLOT_FORMATS:
        out_path = Path(OUTPUT_DIR) / f"rul_vs_cycle_testdev{TEST_DEVICE}.{fmt}"
        fig.savefig(out_path, dpi=150)
        written.append(out_path)
        log.info("saved RUL plot -> %s", out_path)
    plt.close(fig)

    # Auto-open one preview (the PNG if present, else the first format).
    if AUTO_OPEN_PLOTS and written:
        preview = next((p for p in written if p.suffix == ".png"), written[0])
        open_in_viewer(preview)

    return written


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)-8s | %(message)s")
    plot_rul()
