"""
config/config.py
================
Single source of truth for the entire project.

Reproduces the methodology of:
  Lu, Guo, Liu & Shi (2023), "Remaining useful lifetime estimation for discrete
  power electronic devices using physics-informed neural network",
  Scientific Reports 13:10167. https://doi.org/10.1038/s41598-023-37154-5

Edit ONLY this file to change paths, devices, the train/test split, the
preprocessing parameters, the model, or the training setup. Every other module
imports its settings from here.

Quick map
---------
  Change the held-out test device   -> TEST_DEVICE
  Change which signals feed the net -> INPUT_FEATURES
  Tune the 3-step preprocessing     -> DOWNSAMPLE_WINDOW / STANDARDIZE / EMA_SPAN
  Swap RNN <-> LSTM                 -> BACKBONE
  Tune the physics-informed loss    -> ALPHA / BETA / GAMMA
  Tune training                     -> the Training section
"""
from pathlib import Path

# ============================================================================
# Paths
# ============================================================================
ROOT       = Path(__file__).resolve().parent.parent
DATA_DIR   = ROOT / "data"                  # the NASA .mat files live here
OUTPUT_DIR = ROOT / "outputs"
CKPT_DIR   = OUTPUT_DIR / "checkpoints"

# Per-run bookkeeping (see utils/run_registry.py).
#   RUNS_DIR     -- each training run gets a self-contained sub-folder here,
#                   named after its hyperparameters (model.pt + run_info.json).
#   REGISTRY_CSV -- master log; every training run appends one row.
RUNS_DIR     = OUTPUT_DIR / "runs"
REGISTRY_CSV = OUTPUT_DIR / "runs_registry.csv"

# MLflow needs a proper file:// URI, not a bare path. On Windows a raw path
# like "D:\..." is misread as a URI scheme ("d:"), so .as_uri() is required --
# it builds a valid file:/// URI and URL-encodes spaces in the path.
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)   # must exist before as_uri()
MLFLOW_URI = OUTPUT_DIR.joinpath("mlruns").as_uri()

# ============================================================================
# Reproducibility
# ============================================================================
SEED = 42

# DETERMINISTIC forces bit-for-bit reproducible runs (cuDNN deterministic mode
# + deterministic algorithm selection). It costs some GPU speed and disables
# the cuDNN autotuner, so leave it False for normal training; set True only
# when an exactly reproducible run is needed.
DETERMINISTIC: bool = False

# ============================================================================
# Graphviz  (optional -- only for utils/visualize_model.py view 3)
# ============================================================================
# On Windows the Python graphviz library does not always pick up the `dot`
# executable from PATH, even when `dot -V` works in the shell. Setting the bin
# directory here lets visualize_model.py inject it into the process PATH so the
# render works reliably. Leave as None to rely on the system PATH instead.
# Typical Windows location: r"C:\Program Files\Graphviz\bin"
GRAPHVIZ_BIN_DIR: str | None = r"C:\Program Files\Graphviz\bin"

# Output format(s) for the computation-graph render. Each entry produces one
# file in outputs/ (e.g. model_graph_lstm.png and model_graph_lstm.pdf).
# PDF is vector -- best for paper figures; PNG is a raster preview.
GRAPHVIZ_FORMATS: list[str] = ["png", "pdf"]

# Extra detail in the computation graph (passed to torchviz.make_dot).
#   GRAPH_SHOW_ATTRS -- show autograd-node attributes on each op.
#   GRAPH_SHOW_SAVED -- show the tensors each op saves for the backward pass.
# Both add detail (and clutter); set False for a cleaner graph.
GRAPH_SHOW_ATTRS: bool = False
GRAPH_SHOW_SAVED: bool = True

# Output format(s) for the RUL-vs-cycle plot (utils/main `plot` command).
# One file per entry is written to outputs/. PDF is vector (good for papers),
# PNG is a raster preview.
PLOT_FORMATS: list[str] = ["png", "pdf"]

# After saving, open the result in the OS default viewer so you don't have to
# dig into outputs/. Only ONE file is opened per artifact -- the PNG preview
# (the PDF stays on disk). Set False for headless / server runs; if opening
# fails for any reason the file is still saved -- opening is just a convenience.
AUTO_OPEN_PLOTS: bool = True

# ============================================================================
# NASA IGBT dataset
# ============================================================================
# Raw filenames have inconsistent spacing ("Device2 1.mat", "Device3  1.mat",
# ...), so files are located with a glob pattern rather than an exact name.
DEVICE_FILE_GLOB: str = r"raw/Device{device}*.mat"

# Only devices 2-5 exist in the dataset; the original numbering is kept.
DEVICES: list[int] = [2, 3, 4, 5]

# Leave-one-device-out split. Edit TEST_DEVICE to change the fold;
# TRAIN_DEVICES is derived automatically.
TEST_DEVICE:   int       = 2                # held out for testing
TRAIN_DEVICES: list[int] = [d for d in DEVICES if d != TEST_DEVICE]

# The 8 scalar fields under measurement.steadyState[i].timeDomain.
RAW_FEATURES: list[str] = [
    "supplyVoltage",
    "node1Voltage",
    "node2Voltage",
    "collectorEmitterCurrent",
    "heatSinkTemperature",
    "packageTemperature",
    "internalTemperature",        # note: this channel looks miscalibrated
    "ambientTemperature",
]

# node2Voltage is the collector-emitter voltage Vce -- the precursor signal
# used to estimate RUL (paper, "The NASA dataset" section).
VCE_FIELD: str = "node2Voltage"

# Signals fed to the network as inputs (a subset of RAW_FEATURES).
# The paper uses Vce only, hence a single input feature.
INPUT_FEATURES: list[str] = ["node2Voltage"]
N_FEATURES: int = len(INPUT_FEATURES)        # derived -- do not edit

# Conceptual target. The loader emits the raw Vce trajectory as `targets`;
# the RUL label (RUL = 1 - t/Nf) is constructed later from `times`.
TARGET_FEATURE: str = "RUL"

# ============================================================================
# Preprocessing  (paper: three ordered steps)
# ============================================================================
# Each .mat file carries one placeholder row with a nan Vce -- drop it.
DROP_NAN_ROWS: bool = True

# Step 1 -- average downsampling: number of consecutive raw steady-state
# samples averaged into one square-wave cycle. ~50 yields ~196-219 cycles per
# device, matching the cycle counts in the paper's Fig. 4.
DOWNSAMPLE_WINDOW: int = 50

# Step 2 -- standardization to zero mean / unit standard deviation.
# Leak-free: the mean/std are fitted on the TRAINING devices only and applied
# unchanged to the test device (see data/loader.py train_val_test_split). This
# mirrors deployment, where a new IGBT receives a fixed, known scaling. The
# fitted statistics are saved to outputs/standardizer_stats.json as a
# deployment artifact. Nothing to tune here -- the flag just toggles the step.
STANDARDIZE: bool = True

# Step 3 -- EMA window smoothing. Decay factor theta = 2 / (EMA_SPAN + 1).
EMA_SPAN: int = 30

# ============================================================================
# RUL label construction
# ============================================================================
# End of life is the latch-up failure -- a sharp drop in Vce. The failure cycle
# Nf is the steepest Vce decrease within the final FAILURE_SEARCH_FRACTION of
# the (preprocessed) trajectory. The RUL label is RUL(t) = 1 - t / Nf, so it
# runs from 1.0 down to exactly 0.0 at Nf; cycles after Nf are discarded.
FAILURE_SEARCH_FRACTION: float = 0.25   # scan the last 25% for the failure drop
FAILURE_DROP_THRESHOLD:  float = 0.10   # min |Vce step| (preprocessed units);
                                        # real drops are ~0.3, noise steps ~0.01

# ============================================================================
# Padding
# ============================================================================
# Devices have different lengths; sequences are right-padded into one stacked
# tensor. Padded positions in `times` hold TIME_PAD_SENTINEL, so a device's
# true length is recoverable via (times >= 0).
PAD_VALUE:         float = 0.0
TIME_PAD_SENTINEL: int   = -1

# ============================================================================
# Sequence windowing  (used by the model in later steps)
# ============================================================================
# Many-to-one RNN: SEQ_LEN consecutive values map to one output (paper: s = 10).
SEQ_LEN: int = 10

# How many cycles the window advances between consecutive windows.
# 1 = maximum overlap, every possible window (best for this small dataset);
# >1 = fewer, less-overlapping windows (faster, but discards training pairs).
# Note: with stride > 1, the MDC loss penalises a STRIDE-cycle RUL step rather
# than a 1-cycle step -- still valid, just a coarser monotonicity check.
WINDOW_STRIDE: int = 1

# ============================================================================
# Model architecture
# ============================================================================
# The network is:  recurrent stack  ->  dense head  ->  single RUL output.
# Both parts are fully described here -- models/backbone.py builds whatever
# this section specifies, for either backbone.
#
# Paper-exact architecture (Lu et al. 2023, Fig. 1) is reproduced by:
#   RECURRENT_LAYERS = 1, DENSE_HIDDEN = [10], DENSE_ACTIVATION = ["tanh"]
#
# --- Backbone --------------------------------------------------------------
# "rnn"  -> vanilla RNN  (PI-RNN when PINN_MODE is on)
# "lstm" -> LSTM         (PI-LSTM when PINN_MODE is on)
BACKBONE: str = "lstm"

# --- Recurrent stack -------------------------------------------------------
# All recurrent layers share one hidden size -- PyTorch's nn.RNN / nn.LSTM
# require a uniform width across stacked layers. The recurrent activation is
# fixed by the cell type (RNN = tanh, LSTM = its internal gates) and is not
# configurable -- that is a PyTorch limitation, not a project choice.
RECURRENT_HIDDEN:  int   = 80      # hidden units per recurrent layer (paper: 80)
RECURRENT_LAYERS:  int   = 2       # number of stacked recurrent layers (>= 1)
RECURRENT_DROPOUT: float = 0.0     # dropout BETWEEN recurrent layers;
                                   # ignored by PyTorch when RECURRENT_LAYERS = 1

# --- Dense head ------------------------------------------------------------
# DENSE_HIDDEN lists the HIDDEN layer widths only. The final Linear(-> 1)
# output layer is appended automatically -- the model must emit one RUL value,
# so the output width is not configurable (and cannot be broken by editing).
#
# DENSE_ACTIVATION and DENSE_DROPOUT must each have the SAME length as
# DENSE_HIDDEN (one entry per hidden layer). A mismatch raises a clear error
# at startup. Supported activations: relu, tanh, sigmoid, gelu, leaky_relu,
# elu, identity/none. A dropout of 0.0 means no dropout on that layer.
#
# Example -- a 4-hidden-layer head [10, 4, 3, 2] (output 1 added automatically):
#   DENSE_HIDDEN     = [10, 4, 3, 2]
#   DENSE_ACTIVATION = ["relu", "relu", "relu", "relu"]
#   DENSE_DROPOUT    = [0.0, 0.0, 0.0, 0.0]
DENSE_HIDDEN:     list[int]   = [10]       # paper: one dense layer of 10
DENSE_ACTIVATION: list[str]   = ["tanh"]   # paper: tanh
DENSE_DROPOUT:    list[float] = [0.0]      # paper: no dropout

# ============================================================================
# Data-term loss  (OLS only -- physics terms are NOT changed by this)
# ============================================================================
# The total physics-informed loss is OLS + MDC + BCC (paper Eq. 6). This
# section controls ONLY the OLS data term. MDC and BCC are squared-ReLU
# residuals defined by the paper and are not configurable here.
#
# Options for OLS_LOSS:
#   "mse"       -- mean squared error (paper's choice; the default).
#                  Penalises large errors quadratically. Sensitive to outliers.
#   "mae"       -- mean absolute error (L1). Robust to outliers; treats all
#                  errors linearly. Often preferred when targets have a few
#                  noisy samples.
#   "huber"     -- quadratic near zero, linear in the tails. Best of both.
#                  Knee point set by HUBER_DELTA below.
#   "smooth_l1" -- PyTorch's SmoothL1Loss (Huber with delta fixed at 1.0).
#                  Equivalent to huber when HUBER_DELTA = 1.0.
#   "log_cosh"  -- log(cosh(error)). Smooth everywhere, L1-like for large
#                  errors; uncommon but principled.
#
# Default is "mse" -- changing it changes training dynamics and what the
# model considers a "good" prediction.
OLS_LOSS: str = "mae"

# Huber knee point -- error magnitude below which Huber behaves like MSE,
# above which it behaves like MAE. Used only when OLS_LOSS = "huber".
HUBER_DELTA: float = 1.0

# ============================================================================
# Physics-informed loss
# ============================================================================
# PINN_MODE is the master switch.
#   False -> baseline: loss/loss_fn.py forces ALPHA = BETA = 0, so only the
#            ordinary least-squares (OLS) data term is used. Quick A/B: flip
#            this one flag, no need to touch ALPHA / BETA.
#   True  -> physics-informed: ALPHA / BETA / GAMMA below are used as written.
PINN_MODE: bool = True

# Physics-loss weights -- USED ONLY when PINN_MODE is True (ignored otherwise).
# Total loss (paper Eq. 6):
#   E_PINN = (1 - ALPHA) * OLS  +  ALPHA * GAMMA * MDC  +  BETA * BCC
#   ALPHA  balances the OLS (data) term against the monotonic-decreasing term.
#   GAMMA  rescales MDC onto the same magnitude as OLS (paper fixes it at 0.1).
#   BETA   weights the boundary-condition term independently.
# Paper out-of-sample setting: ALPHA = 0.1, BETA = 100.0, GAMMA = 0.1.
ALPHA: float = 0.1
BETA:  float = 100.0
GAMMA: float = 10

# ============================================================================
# Optimizer
# ============================================================================
# Which optimizer trains the model. Edit OPTIMIZER; the trainer reads the
# matching block below. Default is "adam" -- this is what the paper uses, and
# what existing runs were trained with. Changing it changes training
# dynamics and reproducibility.
#
# Options:
#   "adam"    -- default. The paper's choice. Adaptive, robust for RNN/LSTM.
#   "adamw"   -- Adam with decoupled weight decay; usually a better choice
#                than "adam" when WEIGHT_DECAY > 0.
#   "sgd"     -- with optional momentum.
#   "rmsprop" -- the classical pre-Adam choice for recurrent nets.
#   "lbfgs"   -- second-order quasi-Newton. Used in PDE-style PINNs (Raissi et
#                al.). NOTE: requires a closure-based training loop. The
#                trainer switches to a full-batch path automatically; SGD-style
#                mini-batching is disabled for this optimizer. Slower per step
#                but can converge in fewer iterations on smooth problems.
OPTIMIZER: str = "adamw"

# SGD-specific.
SGD_MOMENTUM: float = 0.9

# LBFGS-specific. max_iter is the inner iteration count PER call to .step();
# history_size is how many past gradient pairs LBFGS keeps for its inverse-
# Hessian approximation. Defaults follow PyTorch's recommendations.
LBFGS_MAX_ITER:     int = 20
LBFGS_HISTORY_SIZE: int = 100

# ============================================================================
# Training
# ============================================================================
# Compute device. "cuda" uses the GPU when available; the trainer falls back
# to "cpu" automatically if CUDA is not present.
DEVICE: str = "cuda"

EPOCHS:        int   = 2000
BATCH_SIZE:    int   = 64
LEARNING_RATE: float = 1e-3       # paper: Adam optimiser
WEIGHT_DECAY:  float = 0.0

# The monotonic-decreasing loss compares each prediction with the previous one,
# so batch order must stay temporal. Keep this False unless the MDC term is off.
SHUFFLE_TRAIN: bool = False

# Validation strategy: training runs for a fixed EPOCHS with no early stopping
# and no best-checkpoint selection. The test device is evaluated each epoch for
# logging only -- it never drives model selection, so the out-of-sample result
# stays honest. (If a 5th .mat file is added later it can serve as a true
# validation device via the leave-one-out machinery.)

# ============================================================================
# Logging & checkpointing
# ============================================================================
EXPERIMENT_NAME:    str  = "igbt-rul-pinn"
LOG_EVERY_N_EPOCHS: int  = 50

# Save the final model after the last epoch. There is no "best" checkpoint by
# design (see the validation note above) -- "best on test" would contaminate
# the out-of-sample evaluation.
SAVE_FINAL_MODEL: bool = True
