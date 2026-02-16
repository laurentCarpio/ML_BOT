#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
eval_stable_setup.py
Reproduit le setup "stable" BTCUSDT :

- Split strict TRAIN→TEST déjà fait (U_train/U_test)
- Filtres:
    pA >= 0.70
    pB >= top10 TRAIN (anti-leak)
- RR = 1.3
- R basé sur y:
    y==1 -> +1.3
    y==-1 -> -1
    y==0 -> 0
- Pas de filtre vol

Outputs:
- sel_train.parquet / sel_test.parquet
- prints stats + monthly TEST
"""

import argparse
from urllib.parse import urlparse
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.fs as pafs


def read_parquet_any(path: str) -> pd.DataFrame:
    if path.startswith("s3://"):
        u = urlparse(path)
        fs = pafs.S3FileSystem()
        s3_path = f"{u.netloc}/{u.path.lstrip('/')}"
        with fs.open_input_file(s3_path) as f:
            return pq.read_table(f).to_pandas()
    return pq.read_table(path).to_pandas()


def write_parquet_any(df: pd.DataFrame, path: str):
    table = pa.Table.from_pandas(df, preserve_index=False)
    if path.startswith("s3://"):
        u = urlparse(path)
        fs = pafs.S3FileSystem()
        s3_path = f"{u.netloc}/{u.path.lstrip('/')}"
        with fs.open_output_stream(s3_path) as f:
            pq.write_table(table, f)
    else:
        pq.write_table(table, path)


def stats_R(x: np.ndarray) -> dict:
    x = x.astype(float)
    n = len(x)
    if n == 0:
        return {"n": 0}

    win = float(np.mean(x > 0))
    loss = float(np.mean(x < 0))
    flat = float(np.mean(x == 0))
    exp = float(np.mean(x))
    gp = float(np.sum(x[x > 0]))
    gl = float(-np.sum(x[x < 0]))
    pf = (gp / gl) if gl > 0 else np.nan

    eq = np.cumsum(x)
    peak = np.maximum.accumulate(eq)
    dd = eq - peak
    mdd = float(np.min(dd)) if len(dd) else np.nan

    return {
        "n": int(n),
        "expectancy_R": exp,
        "sum_R": float(np.sum(x)),
        "win_rate": win,
        "loss_rate": loss,
        "flat_rate": flat,
        "profit_factor": pf,
        "max_drawdown_R": mdd,
    }


def main():
    ap = argparse.ArgumentParser("Eval stable setup (BTC)")
    ap.add_argument("--U-train", required=True)
    ap.add_argument("--U-test", required=True)

    ap.add_argument("--out-sel-train", required=True)
    ap.add_argument("--out-sel-test", required=True)

    ap.add_argument("--pA-min", type=float, default=0.70)
    ap.add_argument("--topk-pB", type=float, default=0.10)
    ap.add_argument("--RR", type=float, default=1.3)
    args = ap.parse_args()

    U_train = read_parquet_any(args.U_train)
    U_test = read_parquet_any(args.U_test)

    # index
    U_train["t0"] = pd.to_datetime(U_train["t0"], utc=True)
    U_test["t0"] = pd.to_datetime(U_test["t0"], utc=True)
    U_train = U_train.sort_values("t0").set_index("t0")
    U_test = U_test.sort_values("t0").set_index("t0")

    # sanity
    for c in ["pA","pB","y"]:
        if c not in U_train.columns:
            raise RuntimeError(f"U_train missing {c}")
        if c not in U_test.columns:
            raise RuntimeError(f"U_test missing {c}")

    # cutoff learned on TRAIN only (anti-leak)
    train_pool = U_train[U_train["pA"] >= args.pA_min].copy()
    cutoff_pB_train = float(train_pool["pB"].quantile(1.0 - args.topk_pB))

    # selection
    sel_train = train_pool[train_pool["pB"] >= cutoff_pB_train].copy()
    sel_test = U_test[(U_test["pA"] >= args.pA_min) & (U_test["pB"] >= cutoff_pB_train)].copy()

    print("cutoff_pB_train:", cutoff_pB_train)
    print("train_pool size:", len(train_pool))
    print("sel_train rows:", len(sel_train), "| coverage vs U_train:", len(sel_train)/len(U_train))
    print("sel_test  rows:", len(sel_test),  "| coverage vs U_test :", len(sel_test)/len(U_test))

    # R from y
    RR = float(args.RR)
    sel_train["R"] = np.where(sel_train["y"].astype(int) == 1, RR,
                        np.where(sel_train["y"].astype(int) == -1, -1.0, 0.0)).astype(float)
    sel_test["R"] = np.where(sel_test["y"].astype(int) == 1, RR,
                       np.where(sel_test["y"].astype(int) == -1, -1.0, 0.0)).astype(float)

    print("\nR counts (TEST selection):")
    print(sel_test["R"].value_counts().sort_index())
    print("mean_R:", float(sel_test["R"].mean()))

    print("\n=== TEST STATS (stable selection) ===")
    print(stats_R(sel_test["R"].to_numpy()))

    # monthly
    sel_test["ym"] = sel_test.index.to_period("M").astype(str)
    monthly = (sel_test.groupby("ym", sort=True)["R"]
                    .apply(lambda s: pd.Series(stats_R(s.to_numpy())))
                    .reset_index())
    print("\n=== MONTHLY TEST (stable selection) ===")
    print(monthly)

    # write
    write_parquet_any(sel_train.reset_index(), args.out_sel_train)
    write_parquet_any(sel_test.reset_index(), args.out_sel_test)


if __name__ == "__main__":
    main()