# IGBT Remaining-Useful-Life Estimation with a Physics-Informed Neural Network

A reproduction and reusable implementation of:

> Lu, Guo, Liu & Shi (2023), *"Remaining useful lifetime estimation for discrete
> power electronic devices using physics-informed neural network"*,
> **Scientific Reports 13:10167**.
> https://doi.org/10.1038/s41598-023-37154-5

The project estimates the **Remaining Useful Lifetime (RUL)** of IGBT power
devices from the NASA Ageing dataset, using a recurrent neural network whose
loss is augmented with two **physics constraints**. It is built so that the
entire experiment is controlled from a single configuration file and driven
from a single command-line entry point.

---

## 1. What the project does

An IGBT (Insulated-Gate Bipolar Transistor) degrades as it is power-cycled and
eventually fails by latch-up. The **collector-emitter voltage (Vce)** drifts as
the device ages and drops sharply at failure. The goal is to read a short
window of Vce history and predict how much life the device has left.

**RUL is normalised**: `RUL(t) = 1 - t / Nf`, where `Nf` is the failure cycle.
RUL therefore runs from `1.0` (brand new) down to `0.0` (dead).

The network is **many-to-one**: it consumes `SEQ_LEN` consecutive Vce values
and outputs one RUL number. Two physics rules are added to the loss:

- **MDC — Monotonic Decreasing Constraint**: RUL must not increase over time.
- **BCC — Boundary Condition Constraint**: RUL must stay within `[0, 1]`.

A model trained with these constraints is the **PINN**; the same network
trained without them is the **baseline**.

---

## 2. Method at a glance

```
 raw .mat            preprocessing                    windowing            model              loss
┌──────────┐   ┌──────────────────────────┐   ┌──────────────────┐   ┌────────────┐   ┌──────────────────┐
│ Vce       │ → │ 1. average downsample    │ → │ slide a SEQ_LEN  │ → │ RNN / LSTM │ → │ OLS              │
│ trace per │   │ 2. standardize (train-   │   │ window; 1 window │   │ + dense    │   │ + MDC (physics)  │
│ device    │   │    fitted, leak-free)    │   │ → 1 RUL label    │   │ head → 1   │   │ + BCC (physics)  │
│           │   │ 3. EMA smoothing         │   │                  │   │ RUL value  │   │                  │
└──────────┘   └──────────────────────────┘   └──────────────────┘   └────────────┘   └──────────────────┘
```

The total loss reproduces the paper's Eq. 6:

```
E_PINN = (1 - ALPHA) * OLS  +  ALPHA * GAMMA * MDC  +  BETA * BCC
```

When `PINN_MODE = False`, `ALPHA` and `BETA` are forced to `0`, so the loss
collapses to plain OLS — the baseline.

---

## 3. Evaluation protocol

**Leave-one-device-out.** The dataset has four devices (2, 3, 4, 5). One device
is held out as the test device (`TEST_DEVICE`); the other three are used for
training. The held-out device is never seen during training — not by the
weights and not by the standardisation statistics — so the test result is a
genuine out-of-sample estimate.

There is **no validation set and no early stopping**. Training runs for a fixed
number of epochs and the final model is saved. The test device is evaluated
every epoch *for logging only* — it never influences model selection, which
keeps the out-of-sample number honest. (If a fifth device file is added later
it can serve as a true validation device.)

---

## 4. Project structure

```
PINN_NASA_AGING_DATA/
├── config/
│   └── config.py          Single source of truth — every tunable setting
├── data/
│   ├── raw/                The NASA .mat files (Device2..5)
│   ├── features.py         Preprocessing: downsample, standardize, smooth, RUL labels
│   └── loader.py           Loads .mat files, builds the leave-one-out split, windows it
├── models/
│   └── backbone.py         Generic RNN / LSTM builder, driven entirely by config
├── physics/
│   └── laws.py             The MDC and BCC physics residuals (+ diagnostics)
├── loss/
│   └── loss_fn.py          Assembles OLS + MDC + BCC into the total loss
├── training/
│   └── trainer.py          The training loop, MLflow logging, run-folder save
├── evaluation/
│   ├── metrics.py          MSE, RMSE, MAE, R², MaxError
│   └── plot_rul.py         Actual-vs-predicted RUL plot (reads outputs/runs/)
├── utils/
│   ├── seed.py             Reproducibility (seeds every RNG)
│   ├── logger.py           Central logging configuration
│   ├── cli_help.py         All command-line guidance text
│   ├── visualize_model.py  Architecture summary, layer table, computation graph
│   ├── run_registry.py     Per-run folder + master CSV registry of all runs
│   └── open_file.py        Opens a saved figure in the OS default viewer
├── main.py                 Command-line entry point (7 sub-commands)
├── requirements.txt
└── README.md
```

### What each file does

| File | Responsibility |
|---|---|
| `config/config.py` | **The single source of truth.** Every path, device, preprocessing parameter, model dimension, loss weight and training setting lives here. Edit only this file. |
| `data/features.py` | The three preprocessing steps (downsample → standardize → EMA smooth) and the RUL-label construction (`1 - t/Nf`, with failure detection). |
| `data/loader.py` | Reads the `.mat` files, applies preprocessing, builds the leak-free leave-one-out split, and slices trajectories into windows. `get_windowed_split()` is the one-call entry point. |
| `models/backbone.py` | Builds the network from an **architecture spec**. RNN or LSTM, any number of recurrent layers, any dense-head shape — all from config. Saved runs embed their own spec so they always reload correctly. |
| `physics/laws.py` | The MDC and BCC residuals as differentiable functions, plus non-differentiable **diagnostics** that count how often each rule is actually violated. |
| `loss/loss_fn.py` | Combines OLS + MDC + BCC with the `ALPHA / BETA / GAMMA` weights. Reads `PINN_MODE` and zeroes the physics terms in baseline mode. The OLS (data) term itself is selectable via `OLS_LOSS` (mse / mae / huber / smooth_l1 / log_cosh). |
| `training/trainer.py` | The fixed-epoch training loop. Logs every loss component and physics diagnostic to MLflow each epoch. At the end, calls `run_registry` to save a self-contained run folder and append a row to the master CSV. |
| `evaluation/metrics.py` | The five regression metrics and an `evaluate()` convenience function. |
| `evaluation/plot_rul.py` | Plots actual RUL against each saved model's predicted RUL, discovering models from `outputs/runs/`. |
| `utils/run_registry.py` | Per-run bookkeeping. Writes each run's self-contained folder (`model.pt` + `run_info.json`) under `outputs/runs/`, and appends a row to `outputs/runs_registry.csv` capturing every hyperparameter and final metric. |
| `utils/*` | Other cross-cutting helpers: seeding, logging, CLI text, model visualisation, file-opening. |
| `main.py` | The command-line interface. Controls *which action* runs; never changes settings. |

---

## 5. Setup

Requires Python 3.12 and (optionally) a CUDA GPU.

```bash
python -m venv venv
venv\Scripts\activate            # Windows
pip install -r requirements.txt
```

Place the four NASA `.mat` files in `data/raw/`. The loader finds them with a
glob pattern, so the inconsistent filename spacing in the dataset is fine.

**Optional — model visualisation.** `utils/visualize_model.py` can render the
computation graph if the **Graphviz system package** is installed
(https://graphviz.org/download/) and on `PATH`. Without it, the text summary
and layer table still work; only the graph render is skipped.

---

## 6. Usage — the command-line interface

Everything runs through `main.py`. Seven sub-commands:

```bash
python -m main train      # train (baseline or PINN, per config.py), then evaluate
python -m main eval       # evaluate the most recent saved run on the test device
python -m main compare    # compare newest baseline vs newest PINN, side by side
python -m main config     # show current key settings + the config line for each
python -m main viz        # visualise the model architecture
python -m main plot       # plot actual vs predicted RUL for every saved run
python -m main help       # explain every config.py parameter, section by section
```

The CLI **never changes settings** — `config.py` is the single source of truth.
The CLI controls the *action*; when a setting needs changing, the CLI tells you
exactly which `config.py` line to edit.

### Typical workflow: baseline vs PINN

```bash
# 1. train the baseline
#    edit config.py -> PINN_MODE = False
python -m main train

# 2. train the PINN
#    edit config.py -> PINN_MODE = True
python -m main train

# 3. compare them
python -m main compare
```

`train` writes each run into its own self-contained folder under
`outputs/runs/`, named after the configuration (e.g.
`lstm_pinn_dev2_seq25_str1_h256_l4_a0p5_b100p0_g10_e2000/`), and appends one
row to `outputs/runs_registry.csv`. `compare` picks the most recent baseline
and the most recent PINN from that registry. Identical configurations
overwrite their own folder; different configurations produce distinct folders
so runs never collide.

To evaluate a specific older run instead of the most recent one:

```bash
python -m main eval --run outputs/runs/<folder_name>
```

---

## 7. Configuration

Every setting is in `config/config.py`, grouped into sections. Run
`python -m main help` for a live, section-by-section description of every
parameter with its current value. The most commonly changed ones:

| Setting | Controls |
|---|---|
| `TEST_DEVICE` | Which device is held out (the leave-one-out fold). |
| `BACKBONE` | `"rnn"` or `"lstm"`. |
| `PINN_MODE` | Master switch: `True` = physics-informed, `False` = baseline. |
| `ALPHA`, `BETA`, `GAMMA` | Physics-loss weights (used only when `PINN_MODE = True`). |
| `RECURRENT_HIDDEN/LAYERS`, `DENSE_HIDDEN/ACTIVATION/DROPOUT` | The full architecture. |
| `SEQ_LEN`, `WINDOW_STRIDE` | Windowing. |
| `OPTIMIZER` | Optimiser choice: `adam` (default), `adamw`, `sgd`, `rmsprop`, `lbfgs`. |
| `OLS_LOSS` | Data-term loss: `mse` (default), `mae`, `huber`, `smooth_l1`, `log_cosh`. |
| `EPOCHS`, `BATCH_SIZE`, `LEARNING_RATE` | Training. |

### The architecture is fully config-driven

`DENSE_HIDDEN`, `DENSE_ACTIVATION` and `DENSE_DROPOUT` are lists — one entry per
dense hidden layer — so the network shape is declared as data:

```python
DENSE_HIDDEN     = [10, 4, 3, 2]                  # 4 hidden layers
DENSE_ACTIVATION = ["relu", "relu", "relu", "relu"]
DENSE_DROPOUT    = [0.0, 0.0, 0.0, 0.0]
```

The final `Linear(-> 1)` output layer is appended automatically. The three
lists must be the same length, or the model build fails with a clear error.

The paper's exact architecture is reproduced by
`RECURRENT_LAYERS = 1`, `DENSE_HIDDEN = [10]`, `DENSE_ACTIVATION = ["tanh"]`.

### Optimiser and data-loss choices

Both the optimiser and the data-fit (OLS) loss are config-driven; the
defaults reproduce the paper.

- `OPTIMIZER` selects how the model is trained: `adam` (default, the paper's
  choice), `adamw` (Adam with decoupled weight decay), `sgd` (with optional
  momentum), `rmsprop`, or `lbfgs`. When `lbfgs` is selected the trainer
  automatically switches to a full-batch path with a closure (LBFGS requires
  it); `BATCH_SIZE` is ignored in that mode. LBFGS is slower per `.step()`
  and works best on smooth problems.
- `OLS_LOSS` selects the data-term loss: `mse` (default, the paper's choice),
  `mae`, `huber` (knee point set by `HUBER_DELTA`), `smooth_l1`, or `log_cosh`.
  The physics terms (MDC, BCC) are **not affected** by this choice — they are
  defined by the paper as squared-ReLU residuals and stay that way.

Two important honest caveats:

1. **Training-loss values are not comparable across `OLS_LOSS` choices** —
   they're different functions, so an MAE training loss of `0.05` is not the
   same scale as an MSE training loss of `0.05`. Compare runs using the
   evaluation metrics in `evaluation/metrics.py` (MSE, RMSE, MAE, R², MaxError),
   which stay fixed and live in the run registry CSV.
2. **Each non-default choice changes reproducibility.** Existing runs were
   trained with `OPTIMIZER = "adam"` and `OLS_LOSS = "mse"`. Changing either
   means new runs are no longer directly comparable to those — that's why
   identical configs share a folder name and different configs produce
   different ones.

---

## 8. Outputs

Everything generated lands in `outputs/` (git-ignored):

- `outputs/runs/{descriptive_name}/` — one **self-contained folder per
  training run**. The folder is named after the run's key hyperparameters,
  so runs are identifiable at a glance. Each folder contains:
    - `model.pt` — the trained weights plus the embedded run spec
      (architecture + data pipeline + loss + training). The model can be
      rebuilt and used without reference to the current `config.py`.
    - `run_info.json` — the same information in human-readable form, plus
      the fitted standardiser stats and the final test-device metrics.
- `outputs/runs_registry.csv` — master log. **Every training run appends one
  row**, with every hyperparameter and final metric as its own column, and
  the path to the run folder. The CSV is a complete history; it is never
  rewritten, so re-training the same configuration adds a fresh row (the
  run folder itself is overwritten, but the registry remembers).
- `outputs/mlruns/` — MLflow tracking data.
- `outputs/standardizer_stats.json` — the train-fitted scaling for the most
  recent split (also embedded into each run for deployment).
- `outputs/rul_vs_cycle_testdev{N}.{png,pdf}` — the RUL plot.
- `outputs/model_graph_{backbone}.{png,pdf}` — the computation graph.

### Viewing MLflow

```bash
mlflow ui --backend-store-uri "outputs/mlruns" --port 5000
```

Open http://127.0.0.1:5000 and select the **`igbt-rul-pinn`** experiment (not
"Default"). Each run logs every loss component and physics diagnostic per
epoch, so the training curves can be inspected and runs compared.

---

## 9. Known limitations and honest notes

These are real and worth understanding before drawing conclusions:

- **Single fold.** Results come from one leave-one-out fold. For a robust
  number, all four folds should be rotated and averaged. One fold is evidence,
  not proof.
- **Linear-degradation assumption.** The label `RUL = 1 - t/Nf` is a straight
  line — it assumes the device degrades linearly. Real IGBT wear can
  accelerate near end of life. The model estimates RUL *under this assumption*.
- **The physics terms can be numerically weak.** With the paper's `GAMMA`, the
  MDC term can be orders of magnitude smaller than the data term on this
  dataset, because overlapping windows produce only tiny monotonicity
  violations. The per-epoch diagnostic line (violation rate, max rise) shows
  exactly how active the constraint is — watch it when tuning `GAMMA` / `ALPHA`.
- **No validation set.** By design (see Section 3). The final-epoch model is
  saved, which is not necessarily the best-on-test model — that is the
  deliberate price of keeping the out-of-sample evaluation uncontaminated.

---

## 10. Reproducibility

`SEED` fixes Python, NumPy and PyTorch RNGs. Setting `DETERMINISTIC = True`
additionally forces deterministic algorithms for bit-for-bit reproducible runs
(at some GPU-speed cost). Recurrent layers may still warn that a fully
deterministic kernel is unavailable — that is a known PyTorch limitation.

---

## 11. Citation

If this implementation is useful, please cite the original paper:

```bibtex
@article{lu2023rul,
  title   = {Remaining useful lifetime estimation for discrete power
             electronic devices using physics-informed neural network},
  author  = {Lu, Zhonghai and Guo, Le and Liu, Mingyong and Shi, Tielin},
  journal = {Scientific Reports},
  volume  = {13},
  number  = {1},
  pages   = {10167},
  year    = {2023},
  doi     = {10.1038/s41598-023-37154-5}
}
```
