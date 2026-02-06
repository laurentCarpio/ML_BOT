#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ml_bot/stageB/score_stageB.py

Reads StageB parquet (debug) files containing:
  - id_t (datetime-like)
  - audit_p_thr_ev0 (float)
  - feature columns (feature_cols_xgb from columns.json)

Scores with a StageB XGBoost model (Booster), then writes "scored" parquet with:
  - stgB_p (float32)
  - stgB_s (float32) = stgB_p - audit_p_thr_ev0
  - stgB_allow (int8) = (stgB_s >= thr_s)
  - stgB_thr_s (float32) constant
  - stgB_model_uri (string)
  - stgB_model_stamp (string) = basename(model_uri) or --model-stamp override

Output layout:
  s3://.../data/stageB/dataset=v1/scored/symbol=BTCUSDT/split={train|val|test}/parquet/part-000000.parquet
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import s3fs
import pyarrow as pa
import pyarrow.parquet as pq
import xgboost as xgb


# -----------------------------
# helpers
# -----------------------------
def stamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")

def parse_json_s3(fs: s3fs.S3FileSystem, uri: str) -> dict:
    with fs.open(uri, "rb") as f:
        return json.load(f)

def list_parquet_parts(fs: s3fs.S3FileSystem, prefix: str) -> List[str]:
    """
    prefix like: s3://bucket/path/to/parquet
    Supports:
      - part-*.parquet
      - BTCUSDT_YYYY-MM_part0000*.parquet (stageB.build_stageB.py)
      - any *.parquet
    """
    p = prefix.rstrip("/")
    base = p.replace("s3://", "")

    patterns = [
        base + "/part-*.parquet",
        base + "/*.parquet",
    ]

    out: List[str] = []
    for pat in patterns:
        out.extend([f"s3://{x}" for x in fs.glob(pat)])

    # unique + sorted
    out = sorted(set(out))
    return out

def ensure_dir_s3(fs: s3fs.S3FileSystem, uri_dir: str):
    if not uri_dir.endswith("/"):
        uri_dir += "/"
    if not fs.exists(uri_dir):
        fs.mkdirs(uri_dir, exist_ok=True)

def write_parquet_s3(fs: s3fs.S3FileSystem, uri: str, df: pd.DataFrame, compression: str = "snappy"):
    table = pa.Table.from_pandas(df, preserve_index=False)
    with fs.open(uri, "wb") as f:
        pq.write_table(table, f, compression=compression)

def load_booster(fs: s3fs.S3FileSystem, uri: str) -> xgb.Booster:
    b = xgb.Booster()
    if uri.startswith("s3://"):
        with fs.open(uri, "rb") as f:
            raw = f.read()
        # load_model accepts bytearray for in-memory
        b.load_model(bytearray(raw))
    else:
        b.load_model(uri)
    return b

def default_model_stamp(model_uri: str) -> str:
    base = model_uri.rstrip("/").split("/")[-1]
    # sanitize a bit for parquet string
    base = re.sub(r"[^A-Za-z0-9_.-]+", "_", base)
    return base

def read_parquet_columns_fast(fs: s3fs.S3FileSystem, uri: str) -> List[str]:
    with fs.open(uri, "rb") as f:
        pf = pq.ParquetFile(f)
        return pf.schema.names


# -----------------------------
# scoring
# -----------------------------
def score_split(
    fs: s3fs.S3FileSystem,
    symbol: str,
    split: str,
    in_parquet_dir: str,
    out_root: str,
    columns_json: str,
    model_uri: str,
    thr_s: float,
    model_stamp: Optional[str],
    max_files: int,
    max_rows_per_file: int,
    seed: int,
    keep_all_cols: bool,
):
    meta = parse_json_s3(fs, columns_json)
    feature_cols = meta.get("feature_cols_xgb") or meta.get("feature_cols_all")
    if not isinstance(feature_cols, list) or not feature_cols:
        raise RuntimeError("columns.json missing feature_cols_xgb/feature_cols_all")

    if model_stamp is None or not str(model_stamp).strip():
        model_stamp = default_model_stamp(model_uri)

    booster = load_booster(fs, model_uri)

    # input files
    files = list_parquet_parts(fs, in_parquet_dir)
    if not files:
        raise FileNotFoundError(f"No parquet files found under {in_parquet_dir}/part-*.parquet")

    if max_files > 0:
        files = files[:max_files]

    out_dir = f"{out_root.rstrip('/')}/symbol={symbol}/split={split}/parquet"
    ensure_dir_s3(fs, out_dir)

    rng = np.random.default_rng(int(seed))

    n_in = 0
    n_out = 0
    n_allow = 0

    for i, uri in enumerate(files):
        # robust read: only cols that exist (avoid mismatch)
        cols_in_file = set(read_parquet_columns_fast(fs, uri))
        need = ["audit_p_thr_ev0"] + list(feature_cols)

        # For output, either keep all columns (read whole file) or minimal + ids
        if keep_all_cols:
            use_cols = None  # read all
        else:
            # keep traceability fields if present
            keep_candidates = ["id_row_id", "id_symbol", "id_t", "id_year", "id_month", "label_A"]
            use_cols = [c for c in (keep_candidates + need) if c in cols_in_file]
            # ensure scoring inputs exist
            miss = [c for c in need if c not in cols_in_file]
            if miss:
                raise RuntimeError(f"[{uri}] missing required columns for scoring: {miss}")

        df = pd.read_parquet(uri, columns=use_cols, engine="pyarrow")

        # dev cap rows
        if max_rows_per_file > 0 and len(df) > max_rows_per_file:
            idx = rng.choice(len(df), size=int(max_rows_per_file), replace=False)
            df = df.iloc[idx].reset_index(drop=True)

        # ensure required cols exist
        miss2 = [c for c in (["audit_p_thr_ev0"] + feature_cols) if c not in df.columns]
        if miss2:
            raise RuntimeError(f"[{uri}] missing required columns after read: {miss2}")

        # build DMatrix
        X = df[feature_cols].to_numpy(dtype=np.float32, copy=False)
        dmat = xgb.DMatrix(X, feature_names=feature_cols)
        p = booster.predict(dmat, output_margin=False)
        p = np.asarray(p, dtype=np.float32)

        p_thr = pd.to_numeric(df["audit_p_thr_ev0"], errors="coerce").to_numpy(np.float32, copy=False)
        s = (p - p_thr).astype(np.float32)
        allow = (s >= np.float32(thr_s))

        # attach columns
        df_out = df.copy()
        df_out["stgB_p"] = p.astype(np.float32)
        df_out["stgB_s"] = s.astype(np.float32)
        df_out["stgB_allow"] = allow.astype(np.int8)

        df_out["stgB_thr_s"] = np.float32(thr_s)
        df_out["stgB_model_uri"] = pd.Series([model_uri] * len(df_out), dtype="string")
        df_out["stgB_model_stamp"] = pd.Series([model_stamp] * len(df_out), dtype="string")

        # write
        out_uri = f"{out_dir}/part-{i:06d}.parquet"
        write_parquet_s3(fs, out_uri, df_out.reset_index(drop=True))

        n_in += int(len(df))
        n_out += int(len(df_out))
        n_allow += int(allow.sum())

        print(f"[score_stageB] split={split} file={i+1}/{len(files)} rows={len(df_out):,} allow={int(allow.sum()):,} -> {out_uri}")

    print(f"[score_stageB] DONE split={split} n_in={n_in:,} n_out={n_out:,} n_allow={n_allow:,} allow_frac={n_allow/max(n_out,1):.6g}")


def main():
    ap = argparse.ArgumentParser("Score StageB parquet parts with XGBoost booster and write scored parquet.")

    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])

    ap.add_argument("--stageb-root", default="s3://tradebot-config-tokyo/data/stageB/dataset=v1")
    ap.add_argument("--columns-json", default="s3://tradebot-config-tokyo/data/stageB/dataset=v1/_meta/columns.json")

    ap.add_argument("--model-uri", default="s3://tradebot-config-tokyo/models/xgb-baseline/stageB/baseline_xgb_stageB_20260203-100728.json")

    ap.add_argument("--thr-s", default="0.397767274")

    ap.add_argument("--model-stamp", default="", help="Optional override; otherwise derived from model-uri basename")

    ap.add_argument("--out-root", default="s3://tradebot-config-tokyo/data/stageB/dataset=v1/scored")

    # dev controls
    ap.add_argument("--max-files", type=int, default=0, help="0 = all files in split")
    ap.add_argument("--max-rows-per-file", type=int, default=0, help="0 = no cap")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--keep-all-cols", action="store_true", default=False,
                    help="If set, reads and writes ALL columns from StageB parquet (bigger I/O). Default writes minimal+ids+label if present.")

    args = ap.parse_args()

    fs = s3fs.S3FileSystem()

    model_stamp = args.model_stamp.strip() or None

    print(json.dumps({
        "timestamp": stamp(),
        "symbol": args.symbol,
        "splits": args.splits,
        "stageb_root": args.stageb_root,
        "columns_json": args.columns_json,
        "model_uri": args.model_uri,
        "thr_s": float(args.thr_s),
        "model_stamp": model_stamp or default_model_stamp(args.model_uri),
        "out_root": args.out_root,
        "max_files": int(args.max_files),
        "max_rows_per_file": int(args.max_rows_per_file),
        "keep_all_cols": bool(args.keep_all_cols),
    }, indent=2))

    for sp in args.splits:
        sp = str(sp).strip()
        in_dir = f"{args.stageb_root.rstrip('/')}/split={sp}/parquet"
        score_split(
            fs=fs,
            symbol=str(args.symbol),
            split=sp,
            in_parquet_dir=in_dir,
            out_root=str(args.out_root),
            columns_json=str(args.columns_json),
            model_uri=str(args.model_uri),
            thr_s=float(args.thr_s),
            model_stamp=model_stamp,
            max_files=int(args.max_files),
            max_rows_per_file=int(args.max_rows_per_file),
            seed=int(args.seed),
            keep_all_cols=bool(args.keep_all_cols),
        )

if __name__ == "__main__":
    main()