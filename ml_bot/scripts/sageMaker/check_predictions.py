#!/usr/bin/env python3
"""
check_predictions.py (patched)

Ajouts:
- --invert-proba : utilise 1 - p_pred
- --flip-labels  : utilise 1 - y
- --align-shift K: décale p_pred de K lignes vs y avant métriques/exports

Le reste (histogramme, calibration, metrics, F1 scan, lag scan) est inchangé.
"""

import argparse
import json
import os
import warnings
from typing import Optional
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    brier_score_loss,
    precision_recall_fscore_support
)

def _infer_csv_engine():
    try:
        import pyarrow  # noqa: F401
        return "pyarrow"
    except Exception:
        return None

def load_df(path: str) -> pd.DataFrame:
    engine = _infer_csv_engine()
    try:
        df = pd.read_csv(path, engine=engine) if engine else pd.read_csv(path)
    except Exception as e:
        print(f"[load] Failed with engine={engine!r}, retrying with default. Error: {e}")
        df = pd.read_csv(path)
    return df

def ensure_columns(df: pd.DataFrame, y_col: str, p_col: str, id_col: Optional[str] = None):
    missing = [c for c in [y_col, p_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Present columns: {list(df.columns)[:20]}...")
    if id_col and id_col not in df.columns:
        print(f"[warn] id_col '{id_col}' not found; ignoring id checks.")
        id_col = None
    return id_col

def validate_types_and_ranges(df: pd.DataFrame, y_col: str, p_col: str):
    y_vals = pd.unique(df[y_col].dropna())
    if not set(np.unique(y_vals)).issubset({0, 1}):
        try:
            df[y_col] = pd.to_numeric(df[y_col])
        except Exception:
            pass
        y_vals = pd.unique(df[y_col].dropna())
        if not set(np.unique(y_vals)).issubset({0, 1}):
            raise ValueError(f"[y] Non-binary values detected: sample={y_vals[:10]}")
    p = df[p_col].astype(float)
    bad_mask = (p < 0) | (p > 1) | ~np.isfinite(p)
    bad_ratio = bad_mask.mean()
    hints = []
    if bad_ratio > 0:
        hints.append(f"{bad_ratio:.2%} of p_pred out of [0,1] or non-finite")
    if p.min() < -5 or p.max() > 5:
        hints.append("p_pred range looks like margins/logits (outside [-5,5]). Did you pass raw scores instead of probabilities?")
    return hints

def plot_histogram(p: pd.Series, out_dir: str, title: str = "Distribution of p_pred"):
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(7,5))
    plt.hist(p.dropna().values, bins=50)
    plt.xlabel("p_pred")
    plt.ylabel("count")
    plt.title(title)
    out_path = os.path.join(out_dir, "histogram_p_pred.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path

def calibration_deciles(y: np.ndarray, p: np.ndarray, out_dir: str):
    df = pd.DataFrame({"y": y, "p": p})
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df["decile"] = pd.qcut(df["p"], q=10, labels=False, duplicates="drop")
    cal = df.groupby("decile").agg(
        n=("y", "size"),
        p_mean=("p", "mean"),
        y_rate=("y", "mean"),
        p_min=("p", "min"),
        p_max=("p", "max")
    ).reset_index()
    out_path = os.path.join(out_dir, "calibration_deciles.csv")
    cal.to_csv(out_path, index=False)
    return out_path, cal

def quick_metrics(y: np.ndarray, p: np.ndarray):
    metrics = {}
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    mask = np.isfinite(p)
    y = y[mask]
    p = p[mask]
    pos_rate = float(y.mean()) if len(y) else float("nan")
    metrics["n"] = int(len(y))
    metrics["positive_rate"] = pos_rate
    try:
        metrics["average_precision"] = float(average_precision_score(y, p))
    except Exception as e:
        metrics["average_precision"] = None
        metrics["average_precision_error"] = str(e)
    try:
        metrics["roc_auc"] = float(roc_auc_score(y, p))
    except Exception as e:
        metrics["roc_auc"] = None
        metrics["roc_auc_error"] = str(e)
    try:
        metrics["brier"] = float(brier_score_loss(y, p))
    except Exception as e:
        metrics["brier"] = None
        metrics["brier_error"] = str(e)
    metrics["p_mean"] = float(np.mean(p)) if len(p) else float("nan")
    metrics["p_std"]  = float(np.std(p)) if len(p) else float("nan")
    metrics["p_min"]  = float(np.min(p)) if len(p) else float("nan")
    metrics["p_max"]  = float(np.max(p)) if len(p) else float("nan")
    qs = np.quantile(p, [0.01, 0.05, 0.5, 0.95, 0.99]) if len(p) else [float("nan")]*5
    metrics["p_q01"], metrics["p_q05"], metrics["p_q50"], metrics["p_q95"], metrics["p_q99"] = map(float, qs)
    return metrics

def f1_scan(y: np.ndarray, p: np.ndarray, out_dir: str):
    y = y.astype(int)
    p = p.astype(float)
    mask = np.isfinite(p)
    y, p = y[mask], p[mask]
    if len(p) == 0:
        return None, None
    thr_list = np.unique(np.quantile(p, np.linspace(0.0, 1.0, 101)))
    rows = []
    best = {"f1": -1, "threshold": None, "precision": None, "recall": None}
    for thr in thr_list:
        y_hat = (p >= thr).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(y, y_hat, average="binary", zero_division=0.0)
        rows.append({"threshold": float(thr), "precision": float(precision), "recall": float(recall), "f1": float(f1)})
        if f1 > best["f1"]:
            best = {"f1": float(f1), "threshold": float(thr), "precision": float(precision), "recall": float(recall)}
    out_path = os.path.join(out_dir, "f1_scan.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return out_path, best

def lag_alignment_scan(y: np.ndarray, p: np.ndarray, out_dir: str, max_lag: int = 3):
    rows = []
    best = {"lag": 0, "roc_auc": -1}
    for lag in range(-max_lag, max_lag + 1):
        if lag == 0:
            p_shift = p
            y_shift = y
        elif lag > 0:
            p_shift = p[:-lag]
            y_shift = y[lag:]
        else:
            p_shift = p[-lag:]
            y_shift = y[:lag]
        if len(p_shift) < 2:
            continue
        try:
            auc = roc_auc_score(y_shift, p_shift)
        except Exception:
            auc = np.nan
        rows.append({"lag": lag, "roc_auc": float(auc) if np.isfinite(auc) else None, "n": int(len(y_shift))})
        if np.isfinite(auc) and auc > best["roc_auc"]:
            best = {"lag": lag, "roc_auc": float(auc)}
    out_path = os.path.join(out_dir, "lag_scan.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return out_path, best

def apply_transforms(y: np.ndarray, p: np.ndarray, invert_proba: bool, flip_labels: bool, align_shift: int):
    if invert_proba:
        p = 1.0 - p
        print("[xform] invert-proba: using 1 - p_pred")
    if flip_labels:
        y = 1 - y
        print("[xform] flip-labels: using 1 - y")
    if align_shift != 0:
        k = align_shift
        if k > 0:
            p = p[:-k]
            y = y[k:]
        else:
            k = -k
            p = p[k:]
            y = y[:-k]
        print(f"[xform] align-shift: applied shift of {align_shift} (after shift, n={len(y)})")
    return y, p

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to the CSV file containing y and p_pred.")
    parser.add_argument("--y-col", default="y", help="Column name for ground truth labels (0/1).")
    parser.add_argument("--p-col", default="p_pred", help="Column name for predicted probabilities of class 1.")
    parser.add_argument("--id-col", default=None, help="Optional ID column for additional checks.")
    parser.add_argument("--out", default="_pred_checks", help="Output directory.")
    # NEW switches
    parser.add_argument("--invert-proba", action="store_true", help="Use 1 - p_pred.")
    parser.add_argument("--flip-labels", action="store_true", help="Use 1 - y.")
    parser.add_argument("--align-shift", type=int, default=0, help="Shift p_pred by K rows relative to y (e.g., -2).")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print(f"[info] Loading: {args.csv}")
    df = load_df(args.csv)
    id_col = ensure_columns(df, args.y_col, args.p_col, args.id_col)

    # keep originals for histogram
    p_orig_series = df[args.p_col].astype(float)

    # arrays for transforms & metrics
    y = df[args.y_col].astype(int).values
    p = df[args.p_col].astype(float).values

    # 1) Column & range validation (on original)
    hints = validate_types_and_ranges(df, args.y_col, args.p_col)
    if hints:
        print("[warn] Potential issues with p_pred:")
        for h in hints:
            print(f"       - {h}")
    else:
        print("[ok] y and p_pred columns look valid.")

    # 2) Histogram (toujours sur les scores bruts lus)
    hist_path = plot_histogram(p_orig_series, args.out)
    print(f"[ok] Saved histogram: {hist_path}")

    # Apply transforms for all downstream metrics
    y_use, p_use = apply_transforms(y, p, args.invert_proba, args.flip_labels, args.align_shift)

    # 3) Metrics
    metrics = quick_metrics(y_use, p_use)
    print("[metrics]")
    for k in ["n","positive_rate","average_precision","roc_auc","brier","p_mean","p_std","p_min","p_max","p_q01","p_q05","p_q50","p_q95","p_q99"]:
        print(f"  - {k}: {metrics.get(k)}")
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # calibration deciles / F1 scan sur les données transformées
    cal_path, _ = calibration_deciles(y_use, p_use, args.out)
    print(f"[ok] Saved calibration deciles: {cal_path}")

    f1_path, best_f1 = f1_scan(y_use, p_use, args.out)
    if best_f1:
        print(f"[ok] F1 scan best at threshold={best_f1['threshold']:.4f} "
              f"=> F1={best_f1['f1']:.4f}, precision={best_f1['precision']:.4f}, recall={best_f1['recall']:.4f}")
        print(f"     Saved: {f1_path}")

    # 4) Lag scan (sur les données transformées — devrait maintenant être max à 0 si alignement corrigé)
    lag_path, best_lag = lag_alignment_scan(y_use, p_use, args.out, max_lag=3)
    print(f"[ok] Saved lag scan: {lag_path}")
    if best_lag["lag"] != 0:
        print(f"[warn] AUC is highest at lag={best_lag['lag']} (roc_auc={best_lag['roc_auc']:.4f}). "
              f"This may indicate a remaining alignment issue.")
    else:
        print("[ok] Best AUC at lag=0 — alignment looks good with current transforms.")

    if id_col:
        dup = df[id_col].duplicated().sum()
        if dup > 0:
            print(f"[warn] {dup} duplicated IDs found in '{id_col}'. Ensure uniqueness if IDs should be unique.")
        else:
            print(f"[ok] No duplicated IDs in '{id_col}'.")

    print(f"\n[done] Outputs written under: {os.path.abspath(args.out)}")

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()