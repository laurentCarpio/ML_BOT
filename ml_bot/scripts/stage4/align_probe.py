#!/usr/bin/env python3
import argparse, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

def best_lag_auc(y, p, max_lag=6):
    best = (0, -1.0)
    for lag in range(-max_lag, max_lag+1):
        if lag == 0:
            y2, p2 = y, p
        elif lag > 0:
            y2, p2 = y[lag:], p[:-lag]
        else:
            y2, p2 = y[:lag], p[-lag:]
        if len(y2) > 10:
            try:
                auc = roc_auc_score(y2, p2)
                if auc > best[1]:
                    best = (lag, auc)
            except Exception:
                pass
    return best

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)  # predictions.csv de infer_xgb.py (avec id_cols)
    ap.add_argument("--by", default="symbol,tf,side_num")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    for c in ["y","p_pred"]:
        if c not in df.columns:
            raise SystemExit(f"missing {c} in csv")

    keys = [c for c in [x.strip() for x in args.by.split(",")] if c and c in df.columns]
    if not keys:
        keys = []

    if keys:
        print(f"[info] probing by groups: {keys}")
        rows = []
        for g, gdf in df.groupby(keys, sort=False):
            y = gdf["y"].values.astype(int)
            p = gdf["p_pred"].values.astype(float)
            lag, auc = best_lag_auc(y, p)
            rows.append({"group": str(g), "n": len(gdf), "best_lag": lag, "auc": auc})
        out = pd.DataFrame(rows).sort_values(["best_lag","auc"], ascending=[True, False])
        print(out.head(20).to_string(index=False))
    else:
        y = df["y"].values.astype(int)
        p = df["p_pred"].values.astype(float)
        lag, auc = best_lag_auc(y, p)
        print(f"[global] best_lag={lag} auc={auc:.4f}")

if __name__ == "__main__":
    main()