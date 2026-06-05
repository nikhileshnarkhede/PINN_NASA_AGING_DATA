#!/usr/bin/env python
"""
four_fold_eval.py
=================
Leave-one-device-out (4-fold) reproduction of the PI-LSTM vs LSTM comparison
from Lu, Guo, Liu & Shi (2023), Sci. Rep. 13:10167, on NASA PCoE Dataset #8.

What it does
------------
For each held-out device in {2, 3, 4, 5} it trains TWO models:
  * a vanilla-LSTM baseline   (PINN_MODE = False)
  * a physics-informed LSTM   (PINN_MODE = True)
= 8 training runs total. Every run is forced to use the Lu-matching config
(see LU_CONFIG below) regardless of what config.py currently holds, so the
result is a faithful reproduction, not an accident of the current settings.

After training it reads the per-run metrics back out of
outputs/runs_registry.csv and writes a comparison table to
outputs/four_fold_comparison.csv containing, per fold and averaged:
  - LSTM-baseline MSE and R^2
  - PI-LSTM MSE and R^2
  - improvement (absolute MSE difference and percent reduction)
  - Lu et al. published PI-LSTM MSE / R^2 and the absolute deviation from them

How it works
------------
config.py is the single source of truth, read at import time, so each fold is
run as a SEPARATE process (`python -m main train`) after rewriting the relevant
config lines. config.py is backed up first and restored at the end (and on
error), so your working config is never lost.

Usage
-----
  python four_fold_eval.py                # full reproduction: 8 x 2000-epoch runs
  python four_fold_eval.py --quick 50     # same 8 runs but 50 epochs (plumbing test)
  python four_fold_eval.py --aggregate-only   # skip training; rebuild CSV from the registry
  python four_fold_eval.py --dry-run      # print the planned runs and exit

CAUTION
-------
A full run trains 8 LSTMs for 2000 epochs each -- this can take a long time on
a single GPU. Do a `--quick 50` pass first to confirm the plumbing, and confirm
ONE fold gives a sane R^2 (~0.9, not negative) before trusting the full table.
The only run previously in the registry was a 10-epoch smoke test with
R^2 = -1.36; that is NOT reproduction quality and this script's forced config
exists precisely to avoid repeating it.
"""
from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config" / "config.py"
REGISTRY_CSV = ROOT / "outputs" / "runs_registry.csv"
OUT_CSV = ROOT / "outputs" / "four_fold_comparison.csv"

TEST_DEVICES = [2, 3, 4, 5]

# ---------------------------------------------------------------------------
# Lu-matching configuration -- forced on EVERY run so the reproduction does not
# depend on whatever config.py currently holds. Values are the literal text
# written into config.py (strings carry their quotes).
#
# Source: Lu et al. 2023 -- "RUL estimation method using RNN" (80-cell single
# recurrent layer, dense [10] tanh, s = 10), "The NASA dataset and
# pre-processing" (EMA span = 15), and the out-of-sample LSTM experiment
# (alpha = 0.1, beta = 100, gamma = 0.1, Adam).
#
# NB: this corrects five settings that currently differ from the paper --
# RECURRENT_LAYERS (2 -> 1), EMA_SPAN (30 -> 15), GAMMA (10 -> 0.1),
# OLS_LOSS (mae -> mse), OPTIMIZER (adamw -> adam).
# ---------------------------------------------------------------------------
LU_CONFIG = {
    "BACKBONE":          '"lstm"',
    "RECURRENT_HIDDEN":  "80",
    "RECURRENT_LAYERS":  "1",
    "RECURRENT_DROPOUT": "0.0",
    "DENSE_HIDDEN":      "[10]",
    "DENSE_ACTIVATION":  '["tanh"]',
    "DENSE_DROPOUT":     "[0.0]",
    "SEQ_LEN":           "10",
    "WINDOW_STRIDE":     "1",
    "DOWNSAMPLE_WINDOW": "50",
    "EMA_SPAN":          "15",
    "OLS_LOSS":          '"mse"',
    "OPTIMIZER":         '"adam"',
    "ALPHA":             "0.1",
    "BETA":              "100.0",
    "GAMMA":             "0.1",
    "EPOCHS":            "2000",
    "SHUFFLE_TRAIN":     "False",
    "WEIGHT_DECAY":      "0.0",
    "LEARNING_RATE":     "1e-3",
}

# ---------------------------------------------------------------------------
# Lu et al. (2023) published PI-LSTM out-of-sample results.
# Source: Figure 10 (alpha = 0.1, beta = 100). The MSE bars are x10^-3 and the
# values are read from the bar labels, so they are slightly less precise than a
# table would be. Case -> test-device mapping is from Figure 7 / Table 2:
#   Case1 = dev5, Case2 = dev4, Case3 = dev3, Case4 = dev2.
# Tuple per device: (lstm_test_mse, pinn_test_mse, lstm_test_r2, pinn_test_r2),
# MSE in absolute units (i.e. the x10^-3 bar value already multiplied out).
# ---------------------------------------------------------------------------
LU_PUBLISHED = {
    2: (9.731e-3, 7.285e-3, 0.821, 0.881),
    3: (2.046e-3, 1.650e-3, 0.975, 0.978),
    4: (6.233e-3, 5.963e-3, 0.889, 0.898),
    5: (4.047e-3, 4.086e-3, 0.958, 0.954),  # the one fold where PINN does not help
}
# Paper-reported 4-fold averages (text, p.12): MSE 5.514e-3 -> 4.746e-3;
# R^2 0.911 -> 0.930 (test).
LU_AVG = (5.514e-3, 4.746e-3, 0.911, 0.930)


def set_config(values: dict) -> None:
    """Rewrite top-level assignments in config.py.

    Each key must match exactly one assignment line (with or without a type
    annotation). Raises if a key is missing or ambiguous so a silent no-op can
    never let a run use the wrong setting.
    """
    text = CONFIG_PATH.read_text(encoding="utf-8")
    for key, val in values.items():
        pattern = rf'(?m)^({re.escape(key)}\s*(?::[^=\n]+)?\s*=\s*)[^\n]*$'
        text, n = re.subn(pattern, rf'\g<1>{val}', text)
        if n != 1:
            raise RuntimeError(
                f"set_config: key {key!r} matched {n} lines in config.py "
                f"(expected exactly 1)")
    CONFIG_PATH.write_text(text, encoding="utf-8")


def run_fold(test_device: int, pinn_mode: bool, epochs: str | None) -> None:
    """Force the Lu config for one fold/mode and run `python -m main train`."""
    cfg = dict(LU_CONFIG)
    cfg["TEST_DEVICE"] = str(test_device)
    cfg["PINN_MODE"] = "True" if pinn_mode else "False"
    if epochs is not None:
        cfg["EPOCHS"] = str(epochs)
    set_config(cfg)
    tag = "PI-LSTM" if pinn_mode else "LSTM-baseline"
    print(f"\n{'=' * 70}\n[four_fold] training {tag}  |  test device {test_device}"
          f"\n{'=' * 70}", flush=True)
    subprocess.run([sys.executable, "-m", "main", "train"], cwd=ROOT, check=True)


def _matches_lu_arch(row: dict) -> bool:
    """True if a registry row came from a Lu-architecture run.

    Matching on the architecture (seq_len = 10, single recurrent layer) is
    enough to exclude stale rows from earlier experiments (e.g. the seq25 /
    4-layer smoke test) without depending on whether alpha/beta/gamma are
    recorded raw or zeroed for the baseline.
    """
    try:
        return int(row["seq_len"]) == 10 and int(row["recurrent_layers"]) == 1
    except (KeyError, TypeError, ValueError):
        return False


def latest_metrics(test_device: int, pinn_mode: bool) -> dict | None:
    """Most recent Lu-architecture registry row for this (lstm, device, mode)."""
    if not REGISTRY_CSV.exists():
        return None
    want = "True" if pinn_mode else "False"
    matches = [
        r for r in csv.DictReader(REGISTRY_CSV.open(encoding="utf-8"))
        if r.get("backbone") == "lstm"
        and r.get("test_device") == str(test_device)
        and r.get("pinn_mode") == want
        and _matches_lu_arch(r)
    ]
    return matches[-1] if matches else None


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _pending_row(device, lu, cols):
    lu_b, lu_p, lu_br, lu_pr = lu
    row = {c: "pending" for c in cols}
    row["test_device"] = device
    row["lu_lstm_mse"] = f"{lu_b:.6f}"
    row["lu_pinn_mse"] = f"{lu_p:.6f}"
    row["lu_pinn_r2"] = f"{lu_pr:.4f}"
    return row


def build_comparison() -> list:
    """Read the registry and write outputs/four_fold_comparison.csv.

    Folds without a matching run are written with 'pending' cells (the Lu
    reference column is still filled), so the file -- and the one-pager built
    from it -- always shows the intended shape.
    """
    cols = ["test_device",
            "lstm_baseline_mse", "lstm_baseline_r2",
            "pinn_lstm_mse", "pinn_lstm_r2",
            "improvement_mse_abs", "improvement_mse_pct",
            "lu_lstm_mse", "lu_pinn_mse", "lu_pinn_r2",
            "deviation_pinn_mse_abs", "deviation_pinn_r2_abs"]
    rows = []
    acc = {"b_mse": 0.0, "p_mse": 0.0, "b_r2": 0.0, "p_r2": 0.0}
    n_ok = 0
    missing = []

    for d in TEST_DEVICES:
        base = latest_metrics(d, False)
        pinn = latest_metrics(d, True)
        lu = LU_PUBLISHED[d]
        lu_b, lu_p, lu_br, lu_pr = lu

        if base is None or pinn is None:
            missing.append(d)
            row = _pending_row(d, lu, cols)
            if base is not None:
                row["lstm_baseline_mse"] = f"{_f(base['mse']):.6f}"
                row["lstm_baseline_r2"] = f"{_f(base['r2']):.4f}"
            if pinn is not None:
                row["pinn_lstm_mse"] = f"{_f(pinn['mse']):.6f}"
                row["pinn_lstm_r2"] = f"{_f(pinn['r2']):.4f}"
            rows.append(row)
            continue

        b_mse, b_r2 = _f(base["mse"]), _f(base["r2"])
        p_mse, p_r2 = _f(pinn["mse"]), _f(pinn["r2"])
        imp_abs = b_mse - p_mse
        imp_pct = (imp_abs / b_mse * 100) if b_mse else float("nan")
        rows.append({
            "test_device": d,
            "lstm_baseline_mse": f"{b_mse:.6f}", "lstm_baseline_r2": f"{b_r2:.4f}",
            "pinn_lstm_mse": f"{p_mse:.6f}", "pinn_lstm_r2": f"{p_r2:.4f}",
            "improvement_mse_abs": f"{imp_abs:.6f}", "improvement_mse_pct": f"{imp_pct:.1f}",
            "lu_lstm_mse": f"{lu_b:.6f}", "lu_pinn_mse": f"{lu_p:.6f}", "lu_pinn_r2": f"{lu_pr:.4f}",
            "deviation_pinn_mse_abs": f"{abs(p_mse - lu_p):.6f}",
            "deviation_pinn_r2_abs": f"{abs(p_r2 - lu_pr):.4f}",
        })
        acc["b_mse"] += b_mse; acc["p_mse"] += p_mse
        acc["b_r2"] += b_r2; acc["p_r2"] += p_r2
        n_ok += 1

    if n_ok == len(TEST_DEVICES):
        b_mse = acc["b_mse"] / n_ok; p_mse = acc["p_mse"] / n_ok
        b_r2 = acc["b_r2"] / n_ok; p_r2 = acc["p_r2"] / n_ok
        imp_abs = b_mse - p_mse
        imp_pct = (imp_abs / b_mse * 100) if b_mse else float("nan")
        rows.append({
            "test_device": "AVG",
            "lstm_baseline_mse": f"{b_mse:.6f}", "lstm_baseline_r2": f"{b_r2:.4f}",
            "pinn_lstm_mse": f"{p_mse:.6f}", "pinn_lstm_r2": f"{p_r2:.4f}",
            "improvement_mse_abs": f"{imp_abs:.6f}", "improvement_mse_pct": f"{imp_pct:.1f}",
            "lu_lstm_mse": f"{LU_AVG[0]:.6f}", "lu_pinn_mse": f"{LU_AVG[1]:.6f}",
            "lu_pinn_r2": f"{LU_AVG[3]:.4f}",
            "deviation_pinn_mse_abs": f"{abs(p_mse - LU_AVG[1]):.6f}",
            "deviation_pinn_r2_abs": f"{abs(p_r2 - LU_AVG[3]):.4f}",
        })
    else:
        rows.append(_pending_row("AVG", LU_AVG, cols))

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"\n[four_fold] wrote {OUT_CSV}")
    if missing:
        print(f"[four_fold] NOTE: folds {missing} had no Lu-config run in the "
              f"registry -> marked 'pending'. Run training (without "
              f"--aggregate-only) to fill them.")
    return rows


def main():
    ap = argparse.ArgumentParser(
        description="4-fold PI-LSTM vs LSTM reproduction of Lu et al. 2023.")
    ap.add_argument("--quick", type=int, metavar="N",
                    help="override EPOCHS with N for a fast plumbing test")
    ap.add_argument("--aggregate-only", action="store_true",
                    help="skip training; rebuild the comparison CSV from the registry")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the planned runs and exit")
    args = ap.parse_args()

    plan = [(d, m) for d in TEST_DEVICES for m in (False, True)]

    if args.dry_run:
        print("[four_fold] planned runs (Lu-matching config forced on each):")
        for d, m in plan:
            print(f"   test device {d}  |  {'PI-LSTM' if m else 'LSTM-baseline'}")
        print(f"[four_fold] EPOCHS per run: {args.quick if args.quick else LU_CONFIG['EPOCHS']}")
        return

    if args.aggregate_only:
        build_comparison()
        return

    epochs = str(args.quick) if args.quick else None
    backup = CONFIG_PATH.parent / (CONFIG_PATH.name + ".fourfold.bak")
    shutil.copy2(CONFIG_PATH, backup)
    print(f"[four_fold] backed up config -> {backup}")
    try:
        for d, m in plan:
            run_fold(d, m, epochs)
    finally:
        shutil.copy2(backup, CONFIG_PATH)
        print(f"[four_fold] restored original config from {backup}")

    build_comparison()


if __name__ == "__main__":
    main()
