"""
main.py
=======
Command-line entry point for the IGBT RUL estimation project.

Seven sub-commands:

  python -m main train      Train a model (baseline or PINN, per config.py),
                            then evaluate it on the test device.
  python -m main eval       Evaluate a saved checkpoint on the test device.
  python -m main compare    Compare the saved baseline and PINN checkpoints
                            side by side on the test device.
  python -m main config     Show current settings and which config.py line
                            controls each.
  python -m main viz        Visualise the model architecture currently
                            described in config.py.
  python -m main plot       Plot actual vs predicted RUL for every saved
                            checkpoint, on the test device.
  python -m main help       Explain every config.py parameter, section by
                            section, with its current value.

Design note -- the CLI never changes settings. config.py is the single source
of truth. To switch between baseline and PINN training you edit
config.PINN_MODE; the CLI tells you exactly which line to change. The CLI
controls the ACTION (verb); config.py controls the SETTINGS.
"""
import argparse
import logging
import sys

import torch

from config.config import (
    BACKBONE, TEST_DEVICE, PINN_MODE, RUNS_DIR, REGISTRY_CSV, DEVICE,
)
from utils.logger import setup_logging
from utils import cli_help

log = logging.getLogger("main")

import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
from warnings import filterwarnings
filterwarnings("ignore")


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def _resolve_device(name: str) -> torch.device:
    """Resolve compute device, falling back to CPU when CUDA is unavailable."""
    if name == "cuda" and not torch.cuda.is_available():
        log.warning("CUDA requested but unavailable; using CPU")
        return torch.device("cpu")
    return torch.device(name)


def _load_run_model(model_path, device: torch.device) -> torch.nn.Module:
    """
    Rebuild a model from a run folder's model.pt and load its weights.

    model.pt embeds the full run spec (saved by the trainer), so the model is
    rebuilt exactly as it was trained -- independent of config.py.

    Args:
        model_path: Path to a run folder's model.pt.
        device:     Compute device to place the model on.

    Returns:
        The model with loaded weights, on `device`, in eval mode.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    from pathlib import Path
    from models.backbone import build_backbone_from_spec

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"model not found: {model_path}")

    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    model = build_backbone_from_spec(ckpt["run_spec"])
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model


def _latest_run(pinn_mode: bool | None = None):
    """
    Return the most recent run from the registry CSV, optionally filtered.

    Args:
        pinn_mode: If given, restrict to runs with this pinn_mode
                   (True = PINN, False = baseline). If None, any run.

    Returns:
        The matching CSV row dict (most recent by timestamp), or None if the
        registry is missing or has no matching run.
    """
    import csv
    from pathlib import Path

    registry = Path(REGISTRY_CSV)
    if not registry.exists():
        return None

    with open(registry, newline="") as f:
        rows = list(csv.DictReader(f))
    if pinn_mode is not None:
        want = str(pinn_mode)
        rows = [r for r in rows if r.get("pinn_mode") == want]
    if not rows:
        return None
    # Rows are appended in chronological order; the last match is the newest.
    return rows[-1]


# ----------------------------------------------------------------------------
# Sub-command handlers
# ----------------------------------------------------------------------------
def cmd_train(_args: argparse.Namespace) -> None:
    """
    `train` -- train per config.PINN_MODE, then evaluate on the test device.
    """
    from training.trainer import train as run_training
    from data.loader import get_windowed_split
    from evaluation.metrics import evaluate

    print(cli_help.mode_banner())

    model, _history = run_training()

    # Evaluate the freshly trained model on the held-out test device.
    _train_w, test_w = get_windowed_split()
    device = _resolve_device(DEVICE)
    metrics = evaluate(model, test_w, device)

    mode = "PINN" if PINN_MODE else "BASELINE"
    print(cli_help.metrics_table(
        f"Test device {TEST_DEVICE}  --  {mode}", metrics))


def cmd_eval(args: argparse.Namespace) -> None:
    """
    `eval` -- evaluate a saved run on the test device.

    With no argument, evaluates the most recent run in the registry. With
    --run, evaluates the given run-folder model.pt.
    """
    from pathlib import Path
    from data.loader import get_windowed_split
    from evaluation.metrics import evaluate

    device = _resolve_device(DEVICE)

    if args.run:
        model_path = Path(args.run)
        if model_path.is_dir():
            model_path = model_path / "model.pt"
    else:
        latest = _latest_run()
        if latest is None:
            log.error("no runs found in the registry %s", REGISTRY_CSV)
            log.error("train a model first:  python -m main train")
            sys.exit(1)
        model_path = Path(latest["model_path"])

    try:
        model = _load_run_model(model_path, device)
    except FileNotFoundError:
        log.error("no model at %s", model_path)
        log.error("train a model first:  python -m main train")
        sys.exit(1)

    _train_w, test_w = get_windowed_split()
    metrics = evaluate(model, test_w, device)

    print(cli_help.metrics_table(
        f"Test device {TEST_DEVICE}  --  run {model_path.parent.name}",
        metrics))


def cmd_compare(_args: argparse.Namespace) -> None:
    """
    `compare` -- compare the most recent baseline run vs the most recent PINN
    run, side by side on the test device.

    The two runs are taken from the registry CSV (newest baseline, newest
    PINN). If either is missing, the baseline-then-PINN workflow is printed.
    """
    from pathlib import Path
    from data.loader import get_windowed_split
    from evaluation.metrics import evaluate

    baseline_run = _latest_run(pinn_mode=False)
    pinn_run     = _latest_run(pinn_mode=True)

    if baseline_run is None or pinn_run is None:
        if baseline_run is None:
            log.error("no baseline run found in the registry")
        if pinn_run is None:
            log.error("no PINN run found in the registry")
        print(cli_help.compare_workflow())
        sys.exit(1)

    device = _resolve_device(DEVICE)
    _train_w, test_w = get_windowed_split()

    log.info("baseline run: %s", Path(baseline_run["folder"]).name)
    log.info("PINN run:     %s", Path(pinn_run["folder"]).name)

    baseline_metrics = evaluate(
        _load_run_model(baseline_run["model_path"], device), test_w, device)
    pinn_metrics = evaluate(
        _load_run_model(pinn_run["model_path"], device), test_w, device)

    print(cli_help.comparison_table(baseline_metrics, pinn_metrics))


def cmd_config(_args: argparse.Namespace) -> None:
    """
    `config` -- print current settings and the config.py line controlling each.
    """
    print(cli_help.settings_overview())


def cmd_viz(_args: argparse.Namespace) -> None:
    """
    `viz` -- visualise the architecture currently described in config.py.

    Delegates to utils.visualize_model.visualize(), which prints the text
    summary and torchinfo table and renders the computation-graph files.
    """
    from utils.visualize_model import visualize
    visualize()


def cmd_plot(_args: argparse.Namespace) -> None:
    """
    `plot` -- plot actual vs predicted RUL for every saved checkpoint.

    Delegates to evaluation.plot_rul.plot_rul(), which draws the actual RUL
    and each checkpoint's prediction against cycle for the test device.
    """
    from evaluation.plot_rul import plot_rul
    plot_rul()


def cmd_help(_args: argparse.Namespace) -> None:
    """
    `help` -- print a section-by-section reference of every config.py parameter.
    """
    print(cli_help.parameter_reference())


# ----------------------------------------------------------------------------
# Argument parser
# ----------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    """
    Build the argparse parser with the four sub-commands.

    Returns:
        A configured ArgumentParser.
    """
    parser = argparse.ArgumentParser(
        prog="main",
        description="IGBT RUL estimation -- train, evaluate and compare "
                    "baseline vs physics-informed (PINN) models.",
        epilog="Switching baseline <-> PINN is done in config.py "
               "(PINN_MODE); run `python -m main config` to see all settings.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True,
                                metavar="{train,eval,compare,config,viz,plot,help}")

    p_train = sub.add_parser(
        "train", help="train a model (baseline or PINN per config.py)")
    p_train.set_defaults(func=cmd_train)

    p_eval = sub.add_parser(
        "eval", help="evaluate a saved run on the test device")
    p_eval.add_argument(
        "--run", type=str, default=None,
        help="path to a run folder or its model.pt "
             "(default: the most recent run in the registry)")
    p_eval.set_defaults(func=cmd_eval)

    p_compare = sub.add_parser(
        "compare", help="compare saved baseline vs PINN checkpoints")
    p_compare.set_defaults(func=cmd_compare)

    p_config = sub.add_parser(
        "config", help="show current settings and how to change them")
    p_config.set_defaults(func=cmd_config)

    p_viz = sub.add_parser(
        "viz", help="visualise the model architecture from config.py")
    p_viz.set_defaults(func=cmd_viz)

    p_plot = sub.add_parser(
        "plot", help="plot actual vs predicted RUL for all saved checkpoints")
    p_plot.set_defaults(func=cmd_plot)

    p_help = sub.add_parser(
        "help", help="explain every config.py parameter, section by section")
    p_help.set_defaults(func=cmd_help)

    return parser


def main() -> None:
    """Parse arguments and dispatch to the selected sub-command."""
    setup_logging(logging.INFO)
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
