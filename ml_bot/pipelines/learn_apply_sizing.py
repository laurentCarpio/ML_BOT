#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
learn_apply_sizing.py

Inputs:
- sel_train.parquet (selection stable) avec pB + R
- sel_test.parquet  (selection stable) avec pB + R

Learn on TRAIN:
- bins de pB (quantiles) sur sel_train
- expectancy par bin
- multipliers monotone croissants (basés sur expectancy, normalisés)

Apply on TEST:
- R_sized = R * mult
- stats + monthly

Outputs:
- sizing_model.json
- sel_test_sized.parquet
"""

import argparse
import json
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


def make_bins_quantile(pb: pd.Series, q_edges: list[float]) -> np.ndarray:
    edges = pb.quantile(q_edges).to_numpy(dtype=float)
    edges = np.unique(edges)
    if len(edges) < 3:
        raise RuntimeError("Degenerate pB edges (too many equal quantiles). Change q_edges.")
    return edges


def learn_monotone_multipliers(sel_train: pd.DataFrame, edges: np.ndarray, min_mult: float, max_mult: float):
    # expectancy by bin on TRAIN
    cats = pd.cut(sel_train["pB"].astype(float), bins=edges, include_lowest=True)
    df = sel_train.copy()
    df["pb_bin"] = cats.astype(str)

    grp = df.groupby("pb_bin", sort=False)["R"].agg(["count", "mean"]).rename(columns={"mean": "expectancy"})
    grp = grp.reset_index()

    # monotone: cumulative max on expectancy (force non-decreasing)
    exp = grp["expectancy"].to_numpy(dtype=float)
    exp_mono = np.maximum.accumulate(exp)

    # normalize to multipliers:
    # mult_i = exp_mono_i / exp_mono_mean  (then clipped)
    denom = float(np.mean(exp_mono)) if np.isfinite(np.mean(exp_mono)) and np.mean(exp_mono) != 0 else 1.0
    mult = exp_mono / denom

    mult = np.clip(mult, min_mult, max_mult)

    grp["expectancy_mono"] = exp_mono
    grp["mult"] = mult

    return grp


def apply_multipliers(df: pd.DataFrame, edges: np.ndarray, mults: np.ndarray) -> pd.DataFrame:
    cats = pd.cut(df["pB"].astype(float), bins=edges, include_lowest=True)
    intervals = list(cats.cat.categories)

    if len(intervals) != len(mults):
        raise RuntimeError(f"bins mismatch: intervals={len(intervals)} mults={len(mults)}")

    mapping = {intervals[i]: float(mults[i]) for i in range(len(intervals))}
    out = df.copy()
    out["pb_bin"] = cats
    out["mult"] = cats.map(mapping).astype(float)
    out["R_base"] = out["R"].astype(float)
    out["R_sized"] = out["R_base"] * out["mult"]
    return out


def monthly(df: pd.DataFrame, col: str) -> pd.DataFrame:
    tmp = df.copy()
    tmp["t0"] = pd.to_datetime(tmp["t0"], utc=True)
    tmp = tmp.sort_values("t0")
    tmp["ym"] = tmp["t0"].dt.to_period("M").astype(str)
    out = (tmp.groupby("ym", sort=True)[col]
              .apply(lambda s: pd.Series(stats_R(s.to_numpy())))
              .reset_index())
    return out


def main():
    ap = argparse.ArgumentParser("Learn + apply monotone sizing")
    ap.add_argument("--sel-train", required=True)
    ap.add_argument("--sel-test", required=True)

    ap.add_argument("--out-model-json", required=True)
    ap.add_argument("--out-test-sized", required=True)

    ap.add_argument("--q-edges", default="0,0.2,0.4,0.6,0.8,1.0")
    ap.add_argument("--min-mult", type=float, default=0.75)
    ap.add_argument("--max-mult", type=float, default=2.00)
    args = ap.parse_args()

    q_edges = [float(x) for x in args.q_edges.split(",")]

    sel_train = read_parquet_any(args.sel_train)
    sel_test = read_parquet_any(args.sel_test)

    for df_name, df in [("sel_train", sel_train), ("sel_test", sel_test)]:
        for c in ["pB", "R", "t0"]:
            if c not in df.columns:
                raise RuntimeError(f"{df_name} missing {c}")

    edges = make_bins_quantile(sel_train["pB"], q_edges=q_edges)

    grp = learn_monotone_multipliers(sel_train, edges, min_mult=args.min_mult, max_mult=args.max_mult)

    # model json
    model = {
        "pb_edges": [float(x) for x in edges.tolist()],
        "q_edges": q_edges,
        "min_mult": float(args.min_mult),
        "max_mult": float(args.max_mult),
        "bins_table": grp.to_dict(orient="records"),
    }
    with open(args.out_model_json, "w", encoding="utf-8") as f:
        json.dump(model, f, indent=2)

    # apply on TEST
    mults = grp["mult"].to_numpy(dtype=float)
    test_sized = apply_multipliers(sel_test, edges, mults)

    print("pB edges learned on TRAIN sel_train:", edges)
    print("\nMult distribution TEST:")
    print(test_sized["mult"].value_counts().sort_index())

    print("\nBASE:", stats_R(test_sized["R_base"].to_numpy()))
    print("SIZED:", stats_R(test_sized["R_sized"].to_numpy()))

    print("\n=== MONTHLY BASELINE ===")
    print(monthly(test_sized, "R_base"))
    print("\n=== MONTHLY SIZED ===")
    print(monthly(test_sized, "R_sized"))

    write_parquet_any(test_sized, args.out_test_sized)


if __name__ == "__main__":
    main()