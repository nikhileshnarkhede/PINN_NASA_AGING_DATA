"""
utils/cli_help.py
=================
User-facing guidance text for the command-line interface (main.py).

This module's single job is to DESCRIBE -- it reads current values from
config.py and formats helpful messages telling the user which config.py line
to edit. It never changes anything; config.py stays the single source of truth.

Keeping these (verbose) strings here keeps main.py short: main.py does the
work, cli_help.py explains it.

Note -- the functions below print config PARAMETER NAMES as literal strings
(e.g. "PINN_MODE"). They are all collected in _PARAM so that if a config
parameter is ever renamed, there is exactly one place here to update.
"""
import warnings
# Suppress all FutureWarnings (common in data science libraries like Pandas)
warnings.simplefilter(action='ignore', category=FutureWarning)
from warnings import filterwarnings
filterwarnings("ignore")

from config.config import (
    PINN_MODE, BACKBONE, TEST_DEVICE, TRAIN_DEVICES,
    ALPHA, BETA, GAMMA, EPOCHS, BATCH_SIZE, LEARNING_RATE,
    SEQ_LEN, WINDOW_STRIDE, DEVICE, DETERMINISTIC, SEED,
)

# Config parameter names, kept in one place. If a parameter is renamed in
# config.py, update it here too.
_PARAM = {
    "pinn_mode": "PINN_MODE",
    "backbone":  "BACKBONE",
    "test_dev":  "TEST_DEVICE",
    "alpha":     "ALPHA",
    "beta":      "BETA",
    "gamma":     "GAMMA",
    "epochs":    "EPOCHS",
}

_RULE = "-" * 70


def mode_banner() -> str:
    """
    Describe the mode the next `train` run will use, and how to switch it.

    Read at call time from config.PINN_MODE, so it always reflects the current
    config. Shown at the start of a `train` run.

    Returns:
        A multi-line string naming the current mode and the exact config.py
        edit needed to switch to the other mode.
    """
    if PINN_MODE:
        current = "PINN (physics-informed)"
        switch_to = "baseline"
        new_value = "False"
    else:
        current = "BASELINE (data-only, no physics)"
        switch_to = "PINN"
        new_value = "True"

    return (
        f"{_RULE}\n"
        f"Training mode: {current}\n"
        f"  (config.py  ->  {_PARAM['pinn_mode']} = {PINN_MODE})\n"
        f"\n"
        f"To train the {switch_to} version instead:\n"
        f"  open config.py and set  {_PARAM['pinn_mode']} = {new_value}\n"
        f"  then run:  python -m main train\n"
        f"{_RULE}"
    )


def compare_workflow() -> str:
    """
    Explain the full baseline-then-PINN-then-compare workflow.

    Shown when `compare` is run but a required checkpoint is missing -- it tells
    the user exactly which config edit and command produces each checkpoint.

    Returns:
        A multi-line workflow instruction string.
    """
    return (
        f"{_RULE}\n"
        f"`compare` needs BOTH a baseline and a PINN checkpoint.\n"
        f"Produce them like this:\n"
        f"\n"
        f"  1. config.py  ->  {_PARAM['pinn_mode']} = False\n"
        f"     python -m main train          (creates the baseline checkpoint)\n"
        f"\n"
        f"  2. config.py  ->  {_PARAM['pinn_mode']} = True\n"
        f"     python -m main train          (creates the PINN checkpoint)\n"
        f"\n"
        f"  3. python -m main compare        (side-by-side comparison)\n"
        f"{_RULE}"
    )


def settings_overview() -> str:
    """
    Format a table of the current key settings and the config.py line for each.

    This is the body of the `config` command -- a one-glance view of the
    project's current setup and how to change each part.

    Returns:
        A multi-line string: every important setting, its current value, and
        the config.py parameter that controls it.
    """
    mode = "PINN" if PINN_MODE else "BASELINE"
    rows = [
        ("Mode",            mode,                    _PARAM["pinn_mode"]),
        ("Backbone",        BACKBONE,                _PARAM["backbone"]),
        ("Test device",     str(TEST_DEVICE),        _PARAM["test_dev"]),
        ("Train devices",   str(TRAIN_DEVICES),      "(derived from TEST_DEVICE)"),
        ("Alpha",           str(ALPHA),              _PARAM["alpha"]),
        ("Beta",            str(BETA),               _PARAM["beta"]),
        ("Gamma",           str(GAMMA),              _PARAM["gamma"]),
        ("Epochs",          str(EPOCHS),             _PARAM["epochs"]),
        ("Batch size",      str(BATCH_SIZE),         "BATCH_SIZE"),
        ("Learning rate",   str(LEARNING_RATE),      "LEARNING_RATE"),
        ("Sequence length", str(SEQ_LEN),            "SEQ_LEN"),
        ("Window stride",   str(WINDOW_STRIDE),      "WINDOW_STRIDE"),
        ("Compute device",  DEVICE,                  "DEVICE"),
        ("Deterministic",   str(DETERMINISTIC),      "DETERMINISTIC"),
        ("Seed",            str(SEED),               "SEED"),
    ]

    lines = [
        _RULE,
        "Current project settings  (edit these in config.py)",
        _RULE,
        f"  {'Setting':<18}{'Value':<22}{'config.py parameter'}",
        f"  {'-' * 18}{'-' * 22}{'-' * 22}",
    ]
    for name, value, param in rows:
        lines.append(f"  {name:<18}{value:<22}{param}")
    lines.append(_RULE)
    lines.append("ALPHA / BETA / GAMMA take effect only when PINN_MODE = True.")
    lines.append(_RULE)
    return "\n".join(lines)


def metrics_table(title: str, metrics: dict[str, float]) -> str:
    """
    Format a single model's evaluation metrics as a small aligned table.

    Args:
        title:   Heading for the table (e.g. "Test device 4 -- BASELINE").
        metrics: Dict from evaluation.metrics.evaluate
                 (keys: mse, rmse, mae, r2, max_error).

    Returns:
        A multi-line formatted string.
    """
    order = [("mse", "MSE"), ("rmse", "RMSE"), ("mae", "MAE"),
             ("r2", "R2"), ("max_error", "Max error")]
    lines = [_RULE, title, _RULE]
    for key, label in order:
        if key in metrics:
            lines.append(f"  {label:<12}{metrics[key]:.6f}")
    lines.append(_RULE)
    return "\n".join(lines)


def comparison_table(
    baseline: dict[str, float],
    pinn: dict[str, float],
) -> str:
    """
    Format a side-by-side baseline vs PINN comparison with improvement.

    "Improvement" is reported so that a positive percentage always means PINN
    is better: for error metrics (MSE/RMSE/MAE/Max error) that is a decrease,
    for R2 it is an increase.

    Args:
        baseline: Metrics dict for the baseline model.
        pinn:     Metrics dict for the PINN model.

    Returns:
        A multi-line formatted comparison table.
    """
    # (key, label, lower_is_better)
    order = [
        ("mse", "MSE", True),
        ("rmse", "RMSE", True),
        ("mae", "MAE", True),
        ("r2", "R2", False),
        ("max_error", "Max error", True),
    ]
    lines = [
        _RULE,
        "Baseline vs PINN  --  test device",
        _RULE,
        f"  {'Metric':<12}{'Baseline':<14}{'PINN':<14}{'Improvement'}",
        f"  {'-' * 12}{'-' * 14}{'-' * 14}{'-' * 14}",
    ]
    for key, label, lower_better in order:
        if key not in baseline or key not in pinn:
            continue
        b, p = baseline[key], pinn[key]
        if lower_better:
            improvement = (b - p) / abs(b) * 100 if b != 0 else 0.0
        else:
            improvement = (p - b) / abs(b) * 100 if b != 0 else 0.0
        lines.append(
            f"  {label:<12}{b:<14.6f}{p:<14.6f}{improvement:+.2f}%")
    lines.append(_RULE)
    lines.append("Positive improvement = PINN is better "
                 "(lower error, or higher R2).")
    lines.append(_RULE)
    return "\n".join(lines)


def parameter_reference() -> str:
    """
    Format a full, section-by-section reference of every config.py parameter.

    This is the body of the `help` command. Each parameter is listed under the
    same section heading used in config.py, with its current value and a
    one-line description of what it controls -- so anyone can learn the whole
    configuration without reading the source file.

    Values are read live from config.py, so the reference always reflects the
    current settings.

    Returns:
        A multi-line reference string.
    """
    # Imported here (not at module top) to keep this large set of names local
    # to the one function that needs them.
    from config.config import (
        ROOT, DATA_DIR, OUTPUT_DIR, CKPT_DIR, MLFLOW_URI,
        SEED, DETERMINISTIC,
        GRAPHVIZ_BIN_DIR, GRAPHVIZ_FORMATS, GRAPH_SHOW_ATTRS, GRAPH_SHOW_SAVED,
        PLOT_FORMATS, AUTO_OPEN_PLOTS,
        DEVICE_FILE_GLOB, DEVICES, TEST_DEVICE, TRAIN_DEVICES,
        RAW_FEATURES, VCE_FIELD, INPUT_FEATURES, N_FEATURES, TARGET_FEATURE,
        DROP_NAN_ROWS, DOWNSAMPLE_WINDOW, STANDARDIZE, EMA_SPAN,
        FAILURE_SEARCH_FRACTION, FAILURE_DROP_THRESHOLD,
        PAD_VALUE, TIME_PAD_SENTINEL,
        SEQ_LEN, WINDOW_STRIDE,
        BACKBONE, RECURRENT_HIDDEN, RECURRENT_LAYERS, RECURRENT_DROPOUT,
        DENSE_HIDDEN, DENSE_ACTIVATION, DENSE_DROPOUT,
        PINN_MODE, ALPHA, BETA, GAMMA,
        DEVICE, EPOCHS, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, SHUFFLE_TRAIN,
        EXPERIMENT_NAME, LOG_EVERY_N_EPOCHS, SAVE_FINAL_MODEL,
    )

    # Each section: (heading, [(param_name, current_value, description), ...]).
    sections = [
        ("Paths", [
            ("ROOT", ROOT, "Project root directory (auto-detected)."),
            ("DATA_DIR", DATA_DIR, "Folder holding the NASA .mat files."),
            ("OUTPUT_DIR", OUTPUT_DIR, "Folder for all generated outputs."),
            ("CKPT_DIR", CKPT_DIR, "Folder where trained checkpoints are saved."),
            ("MLFLOW_URI", MLFLOW_URI, "MLflow tracking location (file:// URI)."),
        ]),
        ("Reproducibility", [
            ("SEED", SEED, "Random seed for Python, NumPy and PyTorch."),
            ("DETERMINISTIC", DETERMINISTIC,
             "Force bit-for-bit reproducible runs (slower; off for normal use)."),
        ]),
        ("Graphviz / visualisation", [
            ("GRAPHVIZ_BIN_DIR", GRAPHVIZ_BIN_DIR,
             "Graphviz bin folder, injected into PATH for the graph render."),
            ("GRAPHVIZ_FORMATS", GRAPHVIZ_FORMATS,
             "File format(s) for the computation-graph render."),
            ("GRAPH_SHOW_ATTRS", GRAPH_SHOW_ATTRS,
             "Show autograd-node attributes in the computation graph."),
            ("GRAPH_SHOW_SAVED", GRAPH_SHOW_SAVED,
             "Show tensors saved for backward in the computation graph."),
            ("PLOT_FORMATS", PLOT_FORMATS,
             "File format(s) for the RUL-vs-cycle plot."),
            ("AUTO_OPEN_PLOTS", AUTO_OPEN_PLOTS,
             "Open the PNG preview automatically after saving a figure."),
        ]),
        ("NASA IGBT dataset", [
            ("DEVICE_FILE_GLOB", DEVICE_FILE_GLOB,
             "Glob pattern used to locate each device's .mat file."),
            ("DEVICES", DEVICES, "All device numbers present in the dataset."),
            ("TEST_DEVICE", TEST_DEVICE,
             "Device held out for testing (the leave-one-out fold)."),
            ("TRAIN_DEVICES", TRAIN_DEVICES,
             "Devices used for training (derived from TEST_DEVICE)."),
            ("RAW_FEATURES", RAW_FEATURES,
             "The 8 scalar fields available in each .mat file."),
            ("VCE_FIELD", VCE_FIELD,
             "Field used as the collector-emitter voltage (failure signal)."),
            ("INPUT_FEATURES", INPUT_FEATURES,
             "Signals fed to the network as inputs."),
            ("N_FEATURES", N_FEATURES,
             "Number of input features (derived -- do not edit)."),
            ("TARGET_FEATURE", TARGET_FEATURE,
             "Conceptual prediction target (RUL)."),
        ]),
        ("Preprocessing", [
            ("DROP_NAN_ROWS", DROP_NAN_ROWS,
             "Drop the placeholder nan row in each .mat file."),
            ("DOWNSAMPLE_WINDOW", DOWNSAMPLE_WINDOW,
             "Raw samples averaged into one cycle (step 1)."),
            ("STANDARDIZE", STANDARDIZE,
             "Standardize to zero mean / unit std, train-fitted (step 2)."),
            ("EMA_SPAN", EMA_SPAN,
             "EMA smoothing span; decay = 2/(span+1) (step 3)."),
        ]),
        ("RUL label construction", [
            ("FAILURE_SEARCH_FRACTION", FAILURE_SEARCH_FRACTION,
             "Fraction of the trajectory's tail scanned for the failure drop."),
            ("FAILURE_DROP_THRESHOLD", FAILURE_DROP_THRESHOLD,
             "Minimum Vce step size counted as the failure."),
        ]),
        ("Padding", [
            ("PAD_VALUE", PAD_VALUE,
             "Value used to right-pad variable-length sequences."),
            ("TIME_PAD_SENTINEL", TIME_PAD_SENTINEL,
             "Marker in `times` for padded positions (true length = times>=0)."),
        ]),
        ("Sequence windowing", [
            ("SEQ_LEN", SEQ_LEN,
             "Window length: consecutive cycles mapped to one RUL output."),
            ("WINDOW_STRIDE", WINDOW_STRIDE,
             "Cycles advanced between consecutive windows (1 = max overlap)."),
        ]),
        ("Model architecture", [
            ("BACKBONE", BACKBONE, "Recurrent cell type: 'rnn' or 'lstm'."),
            ("RECURRENT_HIDDEN", RECURRENT_HIDDEN,
             "Hidden units per recurrent layer."),
            ("RECURRENT_LAYERS", RECURRENT_LAYERS,
             "Number of stacked recurrent layers."),
            ("RECURRENT_DROPOUT", RECURRENT_DROPOUT,
             "Dropout between recurrent layers (needs >1 layer)."),
            ("DENSE_HIDDEN", DENSE_HIDDEN,
             "Dense-head hidden widths (output layer of 1 auto-appended)."),
            ("DENSE_ACTIVATION", DENSE_ACTIVATION,
             "Activation per dense hidden layer (length must match DENSE_HIDDEN)."),
            ("DENSE_DROPOUT", DENSE_DROPOUT,
             "Dropout per dense hidden layer (length must match DENSE_HIDDEN)."),
        ]),
        ("Physics-informed loss", [
            ("PINN_MODE", PINN_MODE,
             "Master switch: True = physics-informed, False = baseline."),
            ("ALPHA", ALPHA,
             "Balances OLS data term vs the monotonic-decreasing term."),
            ("BETA", BETA,
             "Weight of the boundary-condition term."),
            ("GAMMA", GAMMA,
             "Rescales the MDC term onto the same magnitude as OLS."),
        ]),
        ("Training", [
            ("DEVICE", DEVICE,
             "Compute device: 'cuda' or 'cpu' (auto-falls back to cpu)."),
            ("EPOCHS", EPOCHS, "Number of training epochs (fixed; no early stop)."),
            ("BATCH_SIZE", BATCH_SIZE, "Mini-batch size."),
            ("LEARNING_RATE", LEARNING_RATE, "Adam optimiser learning rate."),
            ("WEIGHT_DECAY", WEIGHT_DECAY, "Adam L2 weight-decay coefficient."),
            ("SHUFFLE_TRAIN", SHUFFLE_TRAIN,
             "Shuffle training batches (keep False while MDC is on)."),
        ]),
        ("Logging & checkpointing", [
            ("EXPERIMENT_NAME", EXPERIMENT_NAME, "MLflow experiment name."),
            ("LOG_EVERY_N_EPOCHS", LOG_EVERY_N_EPOCHS,
             "How often a progress line is printed."),
            ("SAVE_FINAL_MODEL", SAVE_FINAL_MODEL,
             "Save the final model after the last epoch."),
        ]),
    ]

    lines = [
        "=" * 78,
        "CONFIG.PY PARAMETER REFERENCE",
        "Every setting lives in config.py -- edit that file, then re-run.",
        "=" * 78,
    ]
    for heading, params in sections:
        lines.append("")
        lines.append(f"[ {heading} ]")
        lines.append("-" * 78)
        for name, value, description in params:
            value_str = str(value)
            if len(value_str) > 30:
                value_str = value_str[:27] + "..."
            lines.append(f"  {name:<24}= {value_str:<32}{description}")
    lines.append("")
    lines.append("=" * 78)
    return "\n".join(lines)
