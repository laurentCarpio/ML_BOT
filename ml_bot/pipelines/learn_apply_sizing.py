#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
learn_apply_sizing.py (NOTEBOOK-MATCH)

Reproduit la logique "bricolage" du notebook:

Inputs:
- sel_train.parquet : sélection TRAIN (stable) avec colonnes ['pB','R'] (+ t0 ou index temps)
- sel_test.parquet  : sélection TEST  (stable) avec colonnes ['pB','R'] (+ t0 ou index temps)

Learn on TRAIN:
1) Bins de pB appris sur TRAIN (quantiles, equal-frequency) via q_edges (ex: 0,0.2,...,1.0)
2) Expectancy par bin sur TRAIN
3) Multipliers basés sur expectancy TRAIN:
   - exp_pos = max(expectancy, 0)
   - scaled = exp_pos / exp_pos.max()
   - mult = 1.0 + (max_mult - 1.0) * scaled
   - clamp [1.0, max_mult]
   (=> boost-only, pas de punition)

Apply on TEST:
- pb_bin assigné via les edges TRAIN
- mult via mapping pb_bin(str) -> mult
- R_base = R
- R_sized = R * mult
- stats globales + monthly

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


# -------------------------
# IO helpers
# -------------------------
def read_parquet_any(path: str) -> pd.DataFrame:
    if path.startswith("s3://"):
        u = urlparse(path)
        fs = pafs.S3FileSystem()
        s3_path = f"{u.netloc}/{u.path.lstrip('/')}"
        with fs.open_input_file(s3_path) as f:
            return pq.read_table(f).to_pandas()
    return pq.read_table(path).to_pandas()


def ensure_t0_col(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a timezone-aware UTC 't0' column exists (used for monthly breakdown)."""
    out = df.copy()
    if "t0" in out.columns:
        out["t0"] = pd.to_datetime(out["t0"], utc=True)
    else:
        out["t0"] = pd.to_datetime(out.index, utc=True)
    return out


def _safe_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    """
    PyArrow ne sait pas caster certains dtype pandas (Interval/Categorical Interval).
    On force ces colonnes en string.
    """
    out = df.copy()
    for c in out.columns:
        s = out[c]
        if isinstance(s.dtype, pd.IntervalDtype):
            out[c] = s.astype(str)
        elif isinstance(s.dtype, pd.CategoricalDtype):
            try:
                cats = s.cat.categories
                if len(cats) > 0 and isinstance(cats[0], pd.Interval):
                    out[c] = s.astype(str)
            except Exception:
                pass
    return out


def write_parquet_any(df: pd.DataFrame, path: str):
    df = _safe_for_parquet(df)
    table = pa.Table.from_pandas(df, preserve_index=False)
    if path.startswith("s3://"):
        u = urlparse(path)
        fs = pafs.S3FileSystem()
        s3_path = f"{u.netloc}/{u.path.lstrip('/')}"
        with fs.open_output_stream(s3_path) as f:
            pq.write_table(table, f)
    else:
        pq.write_table(table, path)


# -------------------------
# Stats
# -------------------------
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


def monthly(df: pd.DataFrame, col: str) -> pd.DataFrame:
    tmp = df.copy()
    tmp["t0"] = pd.to_datetime(tmp["t0"], utc=True)
    tmp = tmp.sort_values("t0")
    # warning tz drop is fine (Period)
    tmp["ym"] = tmp["t0"].dt.to_period("M").astype(str)

    out = (
        tmp.groupby("ym", sort=True)[col]
        .apply(lambda s: pd.Series(stats_R(s.to_numpy())))
        .reset_index()
    )
    return out


# -------------------------
# Binning + sizing (NOTEBOOK logic)
# -------------------------
def make_edges_from_quantiles(pb: pd.Series, q_edges: list[float]) -> np.ndarray:
    """
    Reproduit la logique "bins equal-frequency" : quantiles.
    Equivalent à qcut si q_edges = linspace et pas de duplicates.
    """
    edges = pb.astype(float).quantile(q_edges).to_numpy(dtype=float)
    edges = np.unique(edges)
    if len(edges) < 3:
        raise RuntimeError("Degenerate pB edges (too many equal quantiles). Change q_edges.")
    return edges


def assign_bins(pb: pd.Series, edges: np.ndarray) -> pd.Categorical:
    return pd.cut(pb.astype(float), bins=edges, include_lowest=True)


def learn_multipliers_notebook(sel_train: pd.DataFrame, edges: np.ndarray, max_mult: float) -> pd.DataFrame:
    """
    Notebook:
    - expectancy par bin
    - exp_pos = max(exp, 0)
    - scaled = exp_pos / exp_pos.max()
    - mult = 1 + (max_mult-1) * scaled
    - clip [1, max_mult]
    """
    tmp = sel_train.copy()
    tmp["pb_bin"] = assign_bins(tmp["pB"], edges)

    grp = (
        tmp.groupby("pb_bin", observed=True)["R"]
        .agg(n="count", win_rate=lambda x: (x > 0).mean(), expectancy="mean")
        .reset_index()
    )

    exp = grp["expectancy"].to_numpy(dtype=float)
    exp_pos = np.maximum(exp, 0.0)

    if np.nanmax(exp_pos) > 0:
        scaled = exp_pos / np.nanmax(exp_pos)
    else:
        scaled = np.zeros_like(exp_pos)

    mult = 1.0 + (float(max_mult) - 1.0) * scaled
    mult = np.clip(mult, 1.0, float(max_mult))

    grp["mult"] = mult
    # IMPORTANT: stringify bins to avoid Interval dtype issues + stable mapping
    grp["pb_bin_str"] = grp["pb_bin"].astype(str)
    return grp


def apply_multipliers(sel: pd.DataFrame, edges: np.ndarray, mapping: dict) -> pd.DataFrame:
    out = sel.copy()
    out["pb_bin"] = assign_bins(out["pB"], edges).astype(str)
    out["mult"] = out["pb_bin"].map(mapping).astype(float)

    out["R_base"] = out["R"].astype(float)
    out["R_sized"] = out["R_base"] * out["mult"]
    return out


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser("Learn + apply sizing (notebook logic)")
    ap.add_argument("--sel-train", required=True)
    ap.add_argument("--sel-test", required=True)
    ap.add_argument("--out-model-json", required=True)
    ap.add_argument("--out-test-sized", required=True)

    # keep same interface you already use
    ap.add_argument("--q-edges", default="0,0.2,0.4,0.6,0.8,1.0")
    ap.add_argument("--max-mult", type=float, default=2.0)

    args = ap.parse_args()
    q_edges = [float(x) for x in args.q_edges.split(",")]

    sel_train = ensure_t0_col(read_parquet_any(args.sel_train))
    sel_test = ensure_t0_col(read_parquet_any(args.sel_test))

    for df_name, df in [("sel_train", sel_train), ("sel_test", sel_test)]:
        for c in ["pB", "R", "t0"]:
            if c not in df.columns:
                raise RuntimeError(f"{df_name} missing required column: {c}")

    # 1) edges learned on TRAIN
    edges = make_edges_from_quantiles(sel_train["pB"], q_edges=q_edges)

    # 2) multipliers learned on TRAIN (NOTEBOOK logic)
    bins_table = learn_multipliers_notebook(sel_train, edges=edges, max_mult=args.max_mult)

    mapping = dict(zip(bins_table["pb_bin_str"], bins_table["mult"].astype(float)))

    # 3) apply on TEST
    test_sized = apply_multipliers(sel_test, edges=edges, mapping=mapping)

    print("pB edges learned on TRAIN sel_train:", edges)
    print("\nMult distribution TEST:")
    print(test_sized["mult"].value_counts().sort_index())

    print("\nBASE:", stats_R(test_sized["R_base"].to_numpy()))
    print("SIZED:", stats_R(test_sized["R_sized"].to_numpy()))

    print("\n=== MONTHLY BASELINE ===")
    print(monthly(test_sized, "R_base"))
    print("\n=== MONTHLY SIZED ===")
    print(monthly(test_sized, "R_sized"))

    # 4) write model json (human + machine readable)
    model = {
        "method": "notebook_expectancy_scaled",
        "q_edges": q_edges,
        "pb_edges": [float(x) for x in edges.tolist()],
        "max_mult": float(args.max_mult),
        "bins_table": [
            {
                "pb_bin": row["pb_bin_str"],
                "n": int(row["n"]),
                "win_rate": float(row["win_rate"]),
                "expectancy": float(row["expectancy"]),
                "mult": float(row["mult"]),
            }
            for _, row in bins_table.iterrows()
        ],
    }
    with open(args.out_model_json, "w", encoding="utf-8") as f:
        json.dump(model, f, indent=2)

    # 5) write parquet (safe types)
    write_parquet_any(test_sized, args.out_test_sized)


if __name__ == "__main__":
    main()