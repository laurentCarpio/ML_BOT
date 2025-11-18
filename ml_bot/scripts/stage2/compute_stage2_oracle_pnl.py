#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_stage2_oracle_pnl.py — PnL baseline vs oracle sur Stage2 (train/val/test)

Idée :
  - Baseline "je trade tout" : pour chaque signal
        y_tri = Y ∈ {-1,0,1}
        side_num = +1 (buy), -1 (sell)
        y_dir = y_tri * side_num
        PnL_baseline_bps = +THRESH_BPS si y_dir == +1
                          = -THRESH_BPS si y_dir == -1
                          = 0 sinon
  - Oracle "ML parfait" : ne trade QUE les bons trades ex-post
        PnL_oracle_bps   = +THRESH_BPS si y_dir == +1
                          = 0 sinon

Les PnL sont en bps, agrégés par split. Tu pourras les convertir en USDT par la suite
via ta taille de position effective.
"""

from __future__ import annotations
import argparse, sys
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.fs as pafs

# =========================
# S3 helpers
# =========================

def _fs(region: Optional[str]) -> pafs.S3FileSystem:
    return pafs.S3FileSystem(region=region) if region else pafs.S3FileSystem()

def _strip_s3(uri: str) -> str:
    assert uri.startswith("s3://")
    return uri[len("s3://"):]

def _dataset_from_split(fs: pafs.S3FileSystem, src_root: str, split: str) -> ds.Dataset:
    """
    On copie le pattern de make_stage3_xgb.py :
      root/split/... avec partitioning="hive" si l'arbo a été générée ainsi.
    """
    base = _strip_s3(f"{src_root.rstrip('/')}/{split}")
    return ds.dataset(base, filesystem=fs, format="parquet", partitioning="hive")

# =========================
# Filtres
# =========================

def _optional_filter(schema: ds.Schema,
                     symbols: Optional[List[str]],
                     tfs: Optional[List[str]],
                     years: Optional[List[int]]):
    expr = None

    if symbols and "symbol" in schema.names:
        expr_sym = ds.field("symbol").isin(symbols)
        expr = expr_sym if expr is None else (expr & expr_sym)

    if tfs and "tf" in schema.names:
        expr_tf = ds.field("tf").isin(tfs)
        expr = expr_tf if expr is None else (expr & expr_tf)

    if years and "year" in schema.names:
        expr_yr = ds.field("year").isin(years)
        expr = expr_yr if expr is None else (expr & expr_yr)

    return expr

def _apply_side_choice(expr, schema: ds.Schema, side_choice: str):
    """
    - both      : aucun filtre
    - shortonly : side == 'sell'
    - longonly  : side == 'buy'
    """
    if side_choice == "both":
        return expr
    if "side" not in schema.names:
        raise ValueError("Colonne 'side' absente en Stage2: impossible d'appliquer --side.")
    want = "sell" if side_choice == "shortonly" else "buy"
    side_expr = (ds.field("side") == want)
    return side_expr if expr is None else (expr & side_expr)

# =========================
# PnL logic (en bps)
# =========================

def _compute_pnl_stats(dataset: ds.Dataset,
                       filt,
                       batch_size: int = 200_000) -> Dict[str, float]:
    """
    Retourne un dict avec :
      - n_rows
      - n_pos_dir, n_neg_dir, n_neu_dir
      - pnl_baseline_bps
      - pnl_oracle_bps
    """
    avail = set(dataset.schema.names)
    need = {"Y", "side", "THRESH_BPS"}
    missing = sorted(list(need - avail))
    if missing:
        raise RuntimeError(f"Colonnes manquantes en Stage2 pour le PnL oracle: {missing}")

    proj = ["Y","side","THRESH_BPS"]
    if "symbol" in avail: proj.append("symbol")
    if "tf" in avail:     proj.append("tf")
    if "year" in avail:   proj.append("year")

    scanner = dataset.scanner(columns=proj, filter=filt, batch_size=batch_size)

    total_rows = 0
    n_pos_dir = 0
    n_neg_dir = 0
    n_neu_dir = 0

    pnl_baseline = 0.0
    pnl_oracle   = 0.0

    for batch in scanner.to_batches():
        df = batch.to_pandas()
        if df.empty:
            continue

        # side_num: buy→+1, sell→-1
        side_num = df["side"].map({"buy": 1, "sell": -1}).fillna(0).astype("int8")
        y_tri    = pd.to_numeric(df["Y"], errors="coerce").fillna(0).astype("int8")
        th       = pd.to_numeric(df["THRESH_BPS"], errors="coerce").fillna(0.0).astype("float64")

        y_dir = (y_tri * side_num).astype("int8")

        total_rows += int(len(df))
        n_pos_dir  += int((y_dir ==  1).sum())
        n_neg_dir  += int((y_dir == -1).sum())
        n_neu_dir  += int((y_dir ==  0).sum())

        # Baseline : on trade tous les signaux
        pnl_baseline += float((th[y_dir ==  1]).sum())   # gagnants
        pnl_baseline -= float((th[y_dir == -1]).sum())   # perdants
        # neutres : 0

        # Oracle : on ne prend QUE les signaux gagnants ex-post
        pnl_oracle += float((th[y_dir == 1]).sum())

    return {
        "n_rows": int(total_rows),
        "n_pos_dir": int(n_pos_dir),
        "n_neg_dir": int(n_neg_dir),
        "n_neu_dir": int(n_neu_dir),
        "pnl_baseline_bps": float(pnl_baseline),
        "pnl_oracle_bps": float(pnl_oracle),
    }

# =========================
# CLI
# =========================

def parse_args():
    ap = argparse.ArgumentParser(
        description="Compute baseline vs oracle PnL (en bps) sur Stage2 (train/val/test)."
    )
    ap.add_argument("--src-root", required=True,
                    help="s3://…/data/stage2/v2 (ou équivalent)")
    ap.add_argument("--aws-region", default="ap-northeast-1")
    ap.add_argument("--split", choices=["all","train","val","test"], default="all",
                    help="Par défaut: all (train+val+test).")
    ap.add_argument("--symbols", nargs="*", default=None,
                    help="Optionnel: restreindre à certains symbols (ex: BTCUSDT ETHUSDT).")
    ap.add_argument("--tfs", nargs="*", default=None,
                    help="Optionnel: restreindre à certaines TF (ex: 5m 15m 1h).")
    ap.add_argument("--years", nargs="*", type=int, default=None,
                    help="Optionnel: restreindre à certaines années (si colonne 'year' disponible).")
    ap.add_argument("--batch-size", type=int, default=200_000)
    ap.add_argument("--side", choices=["both","shortonly","longonly"], default="both",
                    help="shortonly→side=='sell', longonly→side=='buy'.")

    return ap.parse_args()

# =========================
# Main
# =========================

def main():
    args = parse_args()
    fs = _fs(args.aws_region)

    splits = ["train","val","test"] if args.split == "all" else [args.split]

    print(f"[config] src_root={args.src_root}")
    print(f"[config] splits={splits}, symbols={args.symbols}, tfs={args.tfs}, years={args.years}, side={args.side}")
    print("")

    results = {}

    for sp in splits:
        print("="*80)
        print(f"🔎 Split: {sp}")
        print("="*80)

        try:
            ds_split = _dataset_from_split(fs, args.src_root, sp)
        except Exception as e:
            print(f"⚠️ Impossible d'ouvrir le split {sp}: {e}")
            continue

        schema = ds_split.schema
        filt = _optional_filter(schema, args.symbols, args.tfs, args.years)
        filt = _apply_side_choice(filt, schema, args.side)

        # Combien de lignes ?
        try:
            n_rows = ds_split.count_rows(filter=filt)
        except Exception:
            # fallback approx. via scan
            n_rows = None

        if n_rows is not None:
            print(f"[info] {sp}: ~{n_rows} lignes après filtres.")
            if n_rows == 0:
                print("⛔ Split vide, skip.")
                continue

        stats = _compute_pnl_stats(ds_split, filt, batch_size=args.batch_size)
        results[sp] = stats

        n_all = max(stats["n_rows"], 1)
        print(f"[rows] total={stats['n_rows']}")
        print(f"      dir>0 (dans le sens trade) : {stats['n_pos_dir']} ({stats['n_pos_dir']/n_all:.2%})")
        print(f"      dir<0 (contre le trade)    : {stats['n_neg_dir']} ({stats['n_neg_dir']/n_all:.2%})")
        print(f"      dir=0 (neutres)            : {stats['n_neu_dir']} ({stats['n_neu_dir']/n_all:.2%})")

        pnl_base = stats["pnl_baseline_bps"]
        pnl_orac = stats["pnl_oracle_bps"]

        print("")
        print(f"PNL baseline (trade TOUT) [bps agg]: {pnl_base:,.2f}")
        print(f"PNL oracle   (trade gagnants only) [bps agg]: {pnl_orac:,.2f}")

        if pnl_orac != 0:
            ratio = pnl_base / pnl_orac
            print(f"→ Ratio baseline / oracle : {ratio:.3f}")
        print("")

    # Résumé global simple
    if results:
        print("="*80)
        print("RÉSUMÉ GLOBAL (bps agrégés)")
        print("="*80)
        for sp, st in results.items():
            print(f"{sp.upper():>6} | rows={st['n_rows']:>8} | "
                  f"pnl_baseline_bps={st['pnl_baseline_bps']:>12.2f} | "
                  f"pnl_oracle_bps={st['pnl_oracle_bps']:>12.2f}")
        print("")
    else:
        print("⚠️ Aucun split valide / non vide traité.")

if __name__ == "__main__":
    main()