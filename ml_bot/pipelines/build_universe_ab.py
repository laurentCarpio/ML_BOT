#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_universe_ab.py
- Load A parquet (pA + entry_px + side)
- Load B parquet (pB + y + dir + side + mfe/mae optionnels)
- Normalize columns
- Build U = A inner-join B on t0
- Optional dir-match filter
- Split strict TRAIN/TEST (time-based)

Outputs:
- U.parquet (full)
- U_train.parquet / U_test.parquet
"""

import argparse
from urllib.parse import urlparse
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.fs as pafs


# -------------------------
# IO
# -------------------------
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


def ensure_dt_index(df: pd.DataFrame, prefer_col: str = "t0") -> pd.DataFrame:
    df = df.copy()
    if prefer_col in df.columns:
        df[prefer_col] = pd.to_datetime(df[prefer_col], utc=True)
        df = df.set_index(prefer_col)
    else:
        df.index = pd.to_datetime(df.index, utc=True)
    return df.sort_index()


def normalize_A(dfA: pd.DataFrame) -> pd.DataFrame:
    dfA = ensure_dt_index(dfA, "t0")

    # rename p->pA
    if "pA" not in dfA.columns and "p" in dfA.columns:
        dfA = dfA.rename(columns={"p": "pA"})

    # rename side->sideA
    if "sideA" not in dfA.columns and "side" in dfA.columns:
        dfA = dfA.rename(columns={"side": "sideA"})

    # ensure dirA
    if "dirA" not in dfA.columns:
        s = dfA["sideA"].astype(str).str.lower()
        dfA["dirA"] = np.where(s.eq("long"), 1, np.where(s.eq("short"), -1, np.nan))

    need = ["pA", "entry_px", "sideA", "dirA"]
    missing = [c for c in need if c not in dfA.columns]
    if missing:
        raise RuntimeError(f"A missing columns: {missing} | cols={list(dfA.columns)}")

    dfA = dfA.dropna(subset=["dirA"]).copy()
    dfA["dirA"] = dfA["dirA"].astype(np.int8)
    dfA["pA"] = dfA["pA"].astype(float)
    dfA["entry_px"] = dfA["entry_px"].astype(float)

    return dfA[need]


def normalize_B(dfB: pd.DataFrame) -> pd.DataFrame:
    dfB = ensure_dt_index(dfB, "t0")

    # rename dir->dirB / side->sideB
    if "dirB" not in dfB.columns and "dir" in dfB.columns:
        dfB = dfB.rename(columns={"dir": "dirB"})
    if "sideB" not in dfB.columns and "side" in dfB.columns:
        dfB = dfB.rename(columns={"side": "sideB"})

    need = ["pB", "dirB", "sideB"]
    missing = [c for c in need if c not in dfB.columns]
    if missing:
        raise RuntimeError(f"B missing columns: {missing} | cols={list(dfB.columns)}")

    keep = ["pB", "dirB", "sideB"]
    for c in ["y", "mfe_R", "mae_R", "R_bps", "t0_close", "tp_rr", "max_pullback_R"]:
        if c in dfB.columns:
            keep.append(c)

    out = dfB[keep].copy()
    out["pB"] = out["pB"].astype(float)
    out["dirB"] = out["dirB"].astype(np.int8)
    return out


def main():
    ap = argparse.ArgumentParser("Build universe U = A ∩ B")
    ap.add_argument("--A", required=True, help="events_all_q70.parquet (or similar) with pA")
    ap.add_argument("--B", required=True, help="events_b_light*.parquet with pB + y")
    ap.add_argument("--out-U", required=True)
    ap.add_argument("--out-train", required=True)
    ap.add_argument("--out-test", required=True)

    ap.add_argument("--train-end", default="2024-12-31")
    ap.add_argument("--test-start", default="2025-01-01")

    ap.add_argument("--dir-match", action="store_true", help="Keep only rows where dirA == dirB")
    args = ap.parse_args()

    A_raw = read_parquet_any(args.A)
    B_raw = read_parquet_any(args.B)

    A = normalize_A(A_raw)
    B = normalize_B(B_raw)

    U = A.join(B, how="inner")
    U["dir_match_AB"] = (U["dirA"].astype(int) == U["dirB"].astype(int))

    if args.dir_match:
        U = U.loc[U["dir_match_AB"]].copy()

    # strict split
    train_end = pd.Timestamp(args.train_end, tz="UTC") + pd.Timedelta(hours=23, minutes=59)
    test_start = pd.Timestamp(args.test_start, tz="UTC")

    U_train = U.loc[U.index <= train_end].copy()
    U_test  = U.loc[U.index >= test_start].copy()

    print("U:", U.shape, "| cols:", list(U.columns))
    print("dir_match rate:", float(U["dir_match_AB"].mean()))
    print("U_train:", U_train.shape, "U_test:", U_test.shape)

    write_parquet_any(U.reset_index().rename(columns={"index": "t0"}), args.out_U)
    write_parquet_any(U_train.reset_index().rename(columns={"index": "t0"}), args.out_train)
    write_parquet_any(U_test.reset_index().rename(columns={"index": "t0"}), args.out_test)


if __name__ == "__main__":
    main()