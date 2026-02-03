#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
stageB.build_stageB.py

Inputs (StageA parts on S3):
  s3://.../data/stageA/symbol=BTCUSDT/year=YYYY/month=MM/part-*.parquet

Outputs (StageB):
  - XGBoost-ready CSV.gz: label + f_* only (numeric)
  - Debug parquet: ids + label + aux + audit/cfg + f_*

Also writes _meta/*:
  columns.json, dtypes.json, feature_groups.json, split_plan.json,
  cleaning_rules.json, data_manifest.json, label_stats.json, stats_train.json

Design:
  - time-based split by month (YYYY-MM)
  - no scaling/normalization (XGBoost handles scale + missing)
  - keep NaNs; replace inf -> NaN
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import fsspec
import pyarrow as pa
import pyarrow.parquet as pq


# -------------------------
# Helpers IO
# -------------------------
def s3_so(region: Optional[str]) -> dict:
    return {"client_kwargs": {"region_name": region}} if region else {}

def ensure_s3_url(p: str) -> str:
    return p if p.startswith("s3://") else f"s3://{p}"

def mkdir(path: str, so: dict):
    if path.startswith("s3://"):
        fs = fsspec.filesystem("s3", **so)
        if not fs.exists(path):
            fs.mkdirs(path, exist_ok=True)
    else:
        os.makedirs(path, exist_ok=True)

def write_json(path: str, obj: dict, so: dict):
    payload = json.dumps(obj, indent=2, ensure_ascii=False).encode("utf-8")
    if path.startswith("s3://"):
        with fsspec.open(path, "wb", **so) as f:
            f.write(payload)
    else:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(payload)

def write_parquet(path: str, df: pd.DataFrame, so: dict, compression: str = "snappy"):
    table = pa.Table.from_pandas(df, preserve_index=False)
    if path.startswith("s3://"):
        with fsspec.open(path, "wb", **so) as f:
            pq.write_table(table, f, compression=compression)
    else:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        pq.write_table(table, path, compression=compression)

def write_csv_gz(path: str, df: pd.DataFrame, so: dict, index: bool = False):
    # pandas can write to fsspec handle
    if path.startswith("s3://"):
        with fsspec.open(path, "wb", **so) as f:
            df.to_csv(f, index=index, compression="gzip")
    else:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=index, compression="gzip")


# -------------------------
# StageA discovery
# -------------------------
MONTH_RE = re.compile(r"/year=(\d{4})/month=(\d{2})/")

def month_key(y: int, m: int) -> str:
    return f"{y:04d}-{m:02d}"

def list_stageA_month_paths(stageA_root: str, symbol: str, so: dict) -> Dict[str, List[str]]:
    """
    Returns dict: { "YYYY-MM": [s3://.../part-00000.parquet, ...] }
    """
    glob_pat = f"{stageA_root.rstrip('/')}/symbol={symbol}/year=*/month=*/part-*.parquet"
    fs, _, paths = fsspec.get_fs_token_paths(glob_pat, storage_options=so)
    paths = [ensure_s3_url(p) for p in sorted(paths)]

    out: Dict[str, List[str]] = {}
    for p in paths:
        m = MONTH_RE.search(p)
        if not m:
            continue
        y = int(m.group(1))
        mo = int(m.group(2))
        k = month_key(y, mo)
        out.setdefault(k, []).append(p)
    return out


# -------------------------
# Split plan
# -------------------------
@dataclass(frozen=True)
class SplitPlan:
    train_start: str
    train_end: str
    val_start: str
    val_end: str
    test_start: str
    test_end: str

def ym_to_int(ym: str) -> int:
    y, m = ym.split("-")
    return int(y) * 100 + int(m)

def in_range(ym: str, start: str, end: str) -> bool:
    v = ym_to_int(ym)
    return ym_to_int(start) <= v <= ym_to_int(end)

def assign_split(ym: str, sp: SplitPlan) -> Optional[str]:
    if in_range(ym, sp.train_start, sp.train_end):
        return "train"
    if in_range(ym, sp.val_start, sp.val_end):
        return "val"
    if in_range(ym, sp.test_start, sp.test_end):
        return "test"
    return None


# -------------------------
# Column selection
# -------------------------
def infer_columns_union(sample_paths: List[str], so: dict, max_files: int = 20) -> List[str]:
    cols = set()
    for p in sample_paths[:max_files]:
        with fsspec.open(p, "rb", **so) as f:
            pf = pq.ParquetFile(f)
            cols.update(pf.schema.names)
    return sorted(cols)

def ensure_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
    return out[cols]

def read_parquet_schema_cols(path: str, so: dict) -> List[str]:
    with fsspec.open(path, "rb", **so) as f:
        pf = pq.ParquetFile(f)
        return pf.schema.names

def align_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    Force df à contenir EXACTEMENT cols (ordre inclus).
    - Ajoute les colonnes manquantes avec NaN
    - Ignore les colonnes extra
    """
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return df[cols]

def build_column_contract(all_cols: List[str], xgb_feature_prefixes: List[str]) -> dict:
    """
    Contract:
      - parquet: ids + label + aux + cfg + audit + ALL features (f_*)
      - csv(xgb): label + ONLY selected feature prefixes (no ids/audit/cfg/aux)
    """
    id_cols = [c for c in ["id_row_id","id_symbol","id_t","id_year","id_month"] if c in all_cols]

    target_cols = [c for c in ["label_A"] if c in all_cols]
    aux_target_cols = [c for c in ["label_A_exit_reason","label_A_reason","label_A_pnl_net_bps","label_A_exit_t_sec"] if c in all_cols]

    audit_cols = [c for c in all_cols if c.startswith("audit_")]
    cfg_cols = [c for c in all_cols if c.startswith("cfg_")]

    # ALL features available
    feat_all = sorted([c for c in all_cols if c.startswith("f_")])

    # XGB subset (training features)
    def keep_for_xgb(c: str) -> bool:
        return any(c.startswith(pref) for pref in xgb_feature_prefixes)

    feat_xgb = [c for c in feat_all if keep_for_xgb(c)]

    csv_cols = (["label_A"] if "label_A" in target_cols else []) + feat_xgb
    parquet_cols = id_cols + target_cols + aux_target_cols + cfg_cols + audit_cols + feat_all

    return {
        "id_cols": id_cols,
        "target_cols": target_cols,
        "aux_target_cols": aux_target_cols,
        "audit_cols": audit_cols,
        "cfg_cols": cfg_cols,
        "feature_cols_all": feat_all,
        "feature_cols_xgb": feat_xgb,
        "csv_cols": csv_cols,
        "parquet_cols": parquet_cols,
        "xgb_feature_prefixes": xgb_feature_prefixes,
    }

# -------------------------
# Cleaning / casting
# -------------------------
def clean_numeric(df: pd.DataFrame) -> pd.DataFrame:
    # replace inf -> NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    return df

def cast_for_xgb(df: pd.DataFrame, label_col: str = "label_A") -> pd.DataFrame:
    """
    Cast:
      - label: int8
      - float columns: float32
      - int columns: int32 (except label)
    Keep NaNs (XGB handles).
    """
    out = df.copy()

    if label_col in out.columns:
        out[label_col] = pd.to_numeric(out[label_col], errors="coerce").fillna(0).astype("int8")

    # cast non-label numeric
    for c in out.columns:
        if c == label_col:
            continue
        if pd.api.types.is_integer_dtype(out[c]):
            out[c] = out[c].astype("int32")
        elif pd.api.types.is_float_dtype(out[c]) or pd.api.types.is_bool_dtype(out[c]):
            out[c] = pd.to_numeric(out[c], errors="coerce").astype("float32")
        else:
            # strings should not exist in CSV; debug parquet can keep them
            pass
    return out


# -------------------------
# Stats
# -------------------------
def quantiles(s: pd.Series, qs=(0.01,0.5,0.99)) -> Dict[str, float]:
    s2 = pd.to_numeric(s, errors="coerce")
    return {f"p{int(q*100):02d}": float(s2.quantile(q)) for q in qs}

def compute_train_stats(df_train: pd.DataFrame, feature_cols: List[str]) -> dict:
    stats = {"generated_utc": datetime.now(timezone.utc).isoformat(), "features": {}}
    for c in feature_cols:
        s = pd.to_numeric(df_train[c], errors="coerce")
        stats["features"][c] = {
            "nan_ratio": float(s.isna().mean()),
            "mean": float(s.mean(skipna=True)) if s.notna().any() else None,
            "std": float(s.std(skipna=True)) if s.notna().any() else None,
            "min": float(s.min(skipna=True)) if s.notna().any() else None,
            "max": float(s.max(skipna=True)) if s.notna().any() else None,
            **quantiles(s, qs=(0.01,0.5,0.99)),
        }
    return stats

def compute_label_stats(df: pd.DataFrame, label_col: str = "label_A") -> dict:
    y = pd.to_numeric(df[label_col], errors="coerce").fillna(0).astype(int)
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "n": int(len(y)),
        "pos": int((y == 1).sum()),
        "neg": int((y == 0).sum()),
        "pos_rate": float((y == 1).mean()),
    }


# -------------------------
# Build StageB
# -------------------------
def build_stageB(
    *,
    stageA_root: str,
    stageB_root: str,
    dataset: str,
    symbol: str,
    aws_region: str,
    split_plan: SplitPlan,
    xgb_feature_prefixes: List[str],
    max_parts_per_month: Optional[int] = None,
):
    so = s3_so(aws_region)

    # Discover StageA
    month_map = list_stageA_month_paths(stageA_root, symbol, so)
    months = sorted(month_map.keys())
    if not months:
        raise RuntimeError(f"No StageA files found under {stageA_root} for symbol={symbol}")
    
    months = [m for m in months if (ym_to_int("2024-01") <= ym_to_int(m) <= ym_to_int("2025-10"))]

    print("months found:", months[:10], "...", months[-3:])
    print("example month -> n parts:", months[0], len(month_map[months[0]]))
    print("example part:", month_map[months[0]][0])

    # Take one file to infer columns
    # --- Infer columns robustly: union over a small sample of StageA files ---
    # union sur plusieurs parts (réduit le risque de contrat incomplet)
    some_paths = []
    for ym in months[:3]:
        some_paths += month_map[ym][:5]
    all_cols = infer_columns_union(some_paths, so, max_files=15)
    contract = build_column_contract(all_cols, xgb_feature_prefixes=xgb_feature_prefixes)

    print("Contract inferred:")
    print("  id_cols:", len(contract["id_cols"]))
    print("  target_cols:", len(contract["target_cols"]))
    print("  aux_target_cols:", len(contract["aux_target_cols"]))
    print("  audit_cols:", len(contract["audit_cols"]))
    print("  cfg_cols:", len(contract["cfg_cols"]))
    print("  csv_cols:", len(contract["csv_cols"]))
    print("  parquet_cols:", len(contract["parquet_cols"]))
    print("  feature_cols_all:", len(contract["feature_cols_all"]))
    print("  feature_cols_xgb:", len(contract["feature_cols_xgb"]))
    print("  csv_cols:", len(contract["csv_cols"]))
    print("  parquet_cols:", len(contract["parquet_cols"]))

    # Output roots
    base_out = f"{stageB_root.rstrip('/')}/dataset={dataset}"
    meta_dir = f"{base_out}/_meta"
    mkdir(meta_dir, so)

    # Meta files (static)
    feature_groups = {
        "groups": {
            "candles": "f_c_",
            "book": "f_b_",
            "trades": "f_t_",
            "cross": "f_x_",
        }
    }
    cleaning_rules = {
        "inf_to_nan": True,
        "fillna": "none",
        "float_dtype": "float32",
        "int_dtype": "int32",
        "csv_contains": ["label_A", "f_*"],
        "parquet_contains": ["id_*","label_*","audit_*","cfg_*","f_*"],
    }
    split_plan_json = {
        "timezone": "UTC",
        "train": {"start": split_plan.train_start, "end": split_plan.train_end},
        "val":   {"start": split_plan.val_start,   "end": split_plan.val_end},
        "test":  {"start": split_plan.test_start,  "end": split_plan.test_end},
    }

    # columns.json (global contract)
    columns_json = {
        "dataset": dataset,
        "symbol": symbol,
        "label_col_for_csv": "label_A",
        "id_cols": contract["id_cols"],
        "target_cols": contract["target_cols"],
        "aux_target_cols": contract["aux_target_cols"],
        "audit_cols": contract["audit_cols"],
        "cfg_cols": contract["cfg_cols"],
        "feature_prefix": "f_",
        "feature_cols_all": contract["feature_cols_all"],
        "feature_cols_xgb": contract["feature_cols_xgb"],
        "xgb_feature_prefixes": contract["xgb_feature_prefixes"],
        "csv_cols": contract["csv_cols"],
        "parquet_cols": contract["parquet_cols"],
    }
    write_json(f"{meta_dir}/columns.json", columns_json, so)

    # columns_xgb.json (explicit XGB schema)
    write_json(f"{meta_dir}/columns_xgb.json", {
        "dataset": dataset,
        "symbol": symbol,
        "label_col": "label_A",
        "feature_cols": contract["feature_cols_xgb"],
        "csv_cols": contract["csv_cols"],
        "xgb_feature_prefixes": contract["xgb_feature_prefixes"],
    }, so)

    # columns_parquet.json (debug schema)
    write_json(f"{meta_dir}/columns_parquet.json", {
        "dataset": dataset,
        "symbol": symbol,
        "parquet_cols": contract["parquet_cols"],
        "feature_cols_all": contract["feature_cols_all"],
    }, so)

    # dtypes.json (best-effort defaults)
    dtypes_json = {}
    for c in contract["csv_cols"]:
        if c == "label_A":
            dtypes_json[c] = "int8"
        else:
            dtypes_json[c] = "float32"

    write_json(f"{meta_dir}/feature_groups.json", feature_groups, so)
    write_json(f"{meta_dir}/cleaning_rules.json", cleaning_rules, so)
    write_json(f"{meta_dir}/split_plan.json", split_plan_json, so)
    write_json(f"{meta_dir}/dtypes.json", dtypes_json, so)

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset,
        "symbol": symbol,
        "stageA_root": stageA_root,
        "stageB_root": stageB_root,
        "months_seen": months,
        "files": [],
    }

    # For stats_train: accumulate only TRAIN (CSV cols)
    train_frames_for_stats = []

    # Write outputs month-by-month
    for ym in months:
        split = assign_split(ym, split_plan)
        if split is None:
            # month outside split windows -> skip
            continue

        paths = month_map[ym]
        if max_parts_per_month is not None:
            paths = paths[:max_parts_per_month]

        out_xgb_dir = f"{base_out}/split={split}/xgb"
        out_pq_dir  = f"{base_out}/split={split}/parquet"
        mkdir(out_xgb_dir, so)
        mkdir(out_pq_dir, so)

        for i, p in enumerate(paths):
            # Read only needed cols for parquet debug (includes strings)
            # --- Read only columns that exist in this StageA file (avoid read_parquet errors) ---
            cols_in_file = read_parquet_schema_cols(p, so)
            want_cols = contract["parquet_cols"]
            use_cols = [c for c in want_cols if c in cols_in_file]

            # Lire seulement les colonnes présentes (évite crash si contrat > fichier)
            # --- Read only columns present in this StageA file, then align to full parquet contract ---
            with fsspec.open(p, "rb", **so) as f:
                pf = pq.ParquetFile(f)
                present_cols = set(pf.schema.names)

            want_pq = contract["parquet_cols"]
            use_cols = [c for c in want_pq if c in present_cols]

            df_pq = pd.read_parquet(p, storage_options=so, columns=use_cols, engine="pyarrow")
            df_pq = clean_numeric(df_pq)

            # align to full contract (adds missing cols as NaN, enforces order)
            df_pq = align_columns(df_pq, want_pq)

            # keep traceability in parquet ONLY (not part of contract)
            df_pq["_src"] = p

            # Debug parquet output
            pq_name = f"{symbol}_{ym}_part{i:05d}.parquet"
            pq_out = f"{out_pq_dir}/{pq_name}"
            write_parquet(pq_out, df_pq, so, compression="snappy")

            # XGB CSV output: label + f_* only (numeric)
            csv_cols = contract["csv_cols"]  # label + feature_cols_xgb ONLY
            df_csv = df_pq[csv_cols].copy()  # df_pq a déjà les NaN si colonne manquante dans un fichier
            df_csv = clean_numeric(df_csv)
            df_csv = cast_for_xgb(df_csv, label_col="label_A")
            df_csv = df_csv[csv_cols]  # enforce exact order

            csv_name = f"{symbol}_{ym}_part{i:05d}.csv.gz"
            csv_out = f"{out_xgb_dir}/{csv_name}"
            write_csv_gz(csv_out, df_csv, so, index=False)

            manifest["files"].append({
                "month": ym,
                "split": split,
                "stageA_part": p,
                "stageB_parquet": pq_out,
                "stageB_csv_gz": csv_out,
                "n_rows": int(len(df_csv)),
            })

            if split == "train":
                # keep small memory footprint: sample for stats if huge
                train_frames_for_stats.append(df_csv)

    # Write manifest
    write_json(f"{meta_dir}/data_manifest.json", manifest, so)

    # Build stats on TRAIN
    if train_frames_for_stats:
        df_train = pd.concat(train_frames_for_stats, ignore_index=True)
        label_stats = compute_label_stats(df_train, label_col="label_A")
        xgb_feature_cols = [c for c in contract["csv_cols"] if c != "label_A"]
        stats_train = compute_train_stats(df_train, xgb_feature_cols)

        write_json(f"{meta_dir}/label_stats.json", label_stats, so)
        write_json(f"{meta_dir}/stats_train.json", stats_train, so)
    else:
        write_json(f"{meta_dir}/label_stats.json", {"error": "no train data"}, so)
        write_json(f"{meta_dir}/stats_train.json", {"error": "no train data"}, so)

    print("StageB done.")
    print(f"Output base: {base_out}")
    print(f"Meta: {meta_dir}")


# -------------------------
# CLI
# -------------------------
def main():
    ap = argparse.ArgumentParser("Build StageB (XGBoost-ready) from StageA monthly parts.")

    ap.add_argument("--dataset", type=str, default="v1")
    ap.add_argument("--symbol", type=str, default="BTCUSDT")

    ap.add_argument("--stageA-root", type=str, default="s3://tradebot-config-tokyo/data/stageA")
    ap.add_argument("--stageB-root", type=str, default="s3://tradebot-config-tokyo/data/stageB")

    ap.add_argument("--aws-region", type=str, default="ap-northeast-1")

    # Default split windows (tweakable)
    ap.add_argument("--train-start", type=str, default="2024-01")
    ap.add_argument("--train-end",   type=str, default="2025-06")
    ap.add_argument("--val-start",   type=str, default="2025-07")
    ap.add_argument("--val-end",     type=str, default="2025-08")
    ap.add_argument("--test-start",  type=str, default="2025-09")
    ap.add_argument("--test-end",    type=str, default="2025-10")

    ap.add_argument("--max-parts-per-month", type=int, default=None,
                    help="Optional: limit number of StageA part files per month (debug).")
    
    ap.add_argument("--xgb-feature-prefixes", type=str, default="f_b_,f_c_",
                    help="Comma-separated feature prefixes to keep in XGB CSV (e.g. 'f_b_,f_c_').")

    args = ap.parse_args()

    sp = SplitPlan(
        train_start=args.train_start, train_end=args.train_end,
        val_start=args.val_start, val_end=args.val_end,
        test_start=args.test_start, test_end=args.test_end,
    )

    xgb_prefixes = [s.strip() for s in args.xgb_feature_prefixes.split(",") if s.strip()]

    build_stageB(
        stageA_root=args.stageA_root,
        stageB_root=args.stageB_root,
        dataset=args.dataset,
        symbol=args.symbol,
        aws_region=args.aws_region,
        split_plan=sp,
        xgb_feature_prefixes=xgb_prefixes,
        max_parts_per_month=args.max_parts_per_month,
    )

if __name__ == "__main__":
    main()