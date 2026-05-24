"""
utils/visualize_model.py
========================
Visualise the model architecture currently described in config.py.

This is a standalone diagnostic tool -- nothing else in the project imports it.
It builds the model via models.backbone.build_backbone(), so it always shows
exactly the architecture config.py specifies right now: change the config,
re-run, see the new model.

Run with:  python -m utils.visualize_model

Three views, each degrading gracefully by what is installed:

  1. Text summary       -- always available, zero extra dependencies. Prints
                           the recurrent stack and dense head with per-layer
                           shapes and parameter counts.
  2. torchinfo table    -- if `torchinfo` is installed (pip install torchinfo;
                           pure Python, no system dependency). A Keras-style
                           layer table with input/output shapes per layer.
  3. torchviz graph     -- if `torchviz` AND the Graphviz system package are
                           installed. Renders the autograd computation graph
                           to a PNG in outputs/.

If an optional tool is missing the corresponding view is skipped with a clear
message; the text summary still prints. Nothing crashes.
"""
import logging
import os
from pathlib import Path

import torch

from config.config import (
    BACKBONE, N_FEATURES, SEQ_LEN, OUTPUT_DIR, GRAPHVIZ_BIN_DIR,
    GRAPHVIZ_FORMATS, GRAPH_SHOW_ATTRS, GRAPH_SHOW_SAVED, AUTO_OPEN_PLOTS,
    RECURRENT_HIDDEN, RECURRENT_LAYERS, RECURRENT_DROPOUT,
    DENSE_HIDDEN, DENSE_ACTIVATION, DENSE_DROPOUT,
)
from models.backbone import build_backbone, count_parameters
from utils.open_file import open_in_viewer

log = logging.getLogger(__name__)

_RULE = "=" * 70


# ----------------------------------------------------------------------------
# View 1 -- text summary (always available)
# ----------------------------------------------------------------------------
def text_summary(model: torch.nn.Module, backbone_name: str) -> str:
    """
    Build a plain-text summary of the architecture from config + the model.

    Answers "did config.py build the architecture I expected?" with no extra
    dependencies: layer-by-layer widths, activations, dropout, and parameter
    counts split into recurrent stack vs dense head.

    Args:
        model:         The model from build_backbone().
        backbone_name: "rnn" or "lstm".

    Returns:
        A multi-line summary string.
    """
    lines = [
        _RULE,
        f"MODEL ARCHITECTURE  --  backbone '{backbone_name}'",
        _RULE,
        f"Input  : (batch, {SEQ_LEN}, {N_FEATURES})   [seq_len x n_features]",
        "",
        f"Recurrent stack ({backbone_name.upper()}):",
        f"  layers          : {RECURRENT_LAYERS}",
        f"  hidden size     : {RECURRENT_HIDDEN}",
    ]
    dropout_note = ("  (inactive -- needs more than one layer)"
                    if RECURRENT_LAYERS == 1 else "")
    lines.append(f"  inter-dropout   : {RECURRENT_DROPOUT}{dropout_note}")
    lines.append("")
    lines.append("Dense head:")

    prev = RECURRENT_HIDDEN
    for width, activation, dropout in zip(
            DENSE_HIDDEN, DENSE_ACTIVATION, DENSE_DROPOUT):
        drop_str = (f" -> Dropout({dropout})"
                    if dropout and dropout > 0.0 else "")
        lines.append(
            f"  Linear({prev:4d} -> {width:4d}) -> {activation}{drop_str}")
        prev = width
    lines.append(f"  Linear({prev:4d} -> {1:4d})     [output -- auto-appended]")

    lines.append("")
    lines.append("Output : (batch,)   [one RUL value per window]")
    lines.append(_RULE)

    total = count_parameters(model)
    recurrent_params = sum(p.numel() for p in model.recurrent.parameters())
    head_params = sum(p.numel() for p in model.head.parameters())
    lines.append(
        f"Parameters: recurrent = {recurrent_params:,}   "
        f"head = {head_params:,}   total = {total:,}")
    lines.append(_RULE)
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# View 2 -- torchinfo layer table (optional)
# ----------------------------------------------------------------------------
def torchinfo_table(model: torch.nn.Module) -> str | None:
    """
    Build a torchinfo layer table, if the torchinfo package is installed.

    Args:
        model: The model from build_backbone().

    Returns:
        The formatted table string, or None if torchinfo is not installed.
    """
    try:
        from torchinfo import summary
    except ImportError:
        log.info("torchinfo not installed -- skipping the layer table. "
                 "Install it with:  pip install torchinfo")
        return None

    return str(summary(
        model,
        input_size=(1, SEQ_LEN, N_FEATURES),
        col_names=("input_size", "output_size", "num_params"),
        verbose=0,
    ))


# ----------------------------------------------------------------------------
# View 3 -- torchviz computation graph (optional)
# ----------------------------------------------------------------------------
def torchviz_graph(model: torch.nn.Module, output_dir: Path) -> list[Path]:
    """
    Render the autograd computation graph, if torchviz + Graphviz are available.

    One file is written per entry in config.GRAPHVIZ_FORMATS (e.g. PNG and PDF).
    torchviz needs the Graphviz SYSTEM package (a separate, non-pip install).
    If either piece is missing the function logs a clear message and returns an
    empty list rather than raising.

    Args:
        model:      The model from build_backbone().
        output_dir: Directory the rendered files are written to.

    Returns:
        List of written file paths (empty if nothing could be rendered).
    """
    try:
        from torchviz import make_dot
    except ImportError:
        log.info("torchviz not installed -- skipping the computation graph. "
                 "Install it with:  pip install torchviz  "
                 "(also requires the Graphviz system package).")
        return []

    # On Windows the graphviz library does not always find `dot` via PATH even
    # when it works in the shell. If config.GRAPHVIZ_BIN_DIR is set, prepend it
    # to this process's PATH so the child `dot` call inherits it reliably.
    if GRAPHVIZ_BIN_DIR:
        bin_dir = str(GRAPHVIZ_BIN_DIR)
        if Path(bin_dir).is_dir():
            os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
        else:
            log.warning("config.GRAPHVIZ_BIN_DIR is set but not a directory: "
                        "%s -- falling back to system PATH", bin_dir)

    dummy = torch.randn(1, SEQ_LEN, N_FEATURES, requires_grad=True,
                        device=next(model.parameters()).device)
    output = model(dummy)
    graph = make_dot(output.mean(), params=dict(model.named_parameters()),
                     show_attrs=GRAPH_SHOW_ATTRS, show_saved=GRAPH_SHOW_SAVED)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_stem = output_dir / f"model_graph_{BACKBONE}"

    written: list[Path] = []
    for fmt in GRAPHVIZ_FORMATS:
        try:
            # render() invokes the Graphviz `dot` executable; if Graphviz is
            # not installed system-wide this raises, which we catch and report.
            graph.render(str(out_stem), format=fmt, cleanup=True)
        except Exception as exc:                   # graphviz.ExecutableNotFound etc.
            log.warning("could not render '%s' (is the Graphviz system "
                        "package installed and on PATH?): %s", fmt, exc)
            continue
        path = out_stem.with_suffix(f".{fmt}")
        log.info("saved computation graph -> %s", path)
        written.append(path)

    # Auto-open one preview (the PNG if present, else the first format).
    if AUTO_OPEN_PLOTS and written:
        preview = next((p for p in written if p.suffix == ".png"), written[0])
        open_in_viewer(preview)

    return written


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------
def visualize() -> None:
    """
    Build the configured model and print / render every available view.
    """
    model = build_backbone()
    model.eval()

    # View 1 -- always.
    print(text_summary(model, BACKBONE))

    # View 2 -- if torchinfo is installed.
    table = torchinfo_table(model)
    if table is not None:
        print()
        print(table)

    # View 3 -- if torchviz + Graphviz are installed.
    torchviz_graph(model, Path(OUTPUT_DIR))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)-8s | %(message)s")
    visualize()
