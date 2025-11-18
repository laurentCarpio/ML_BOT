#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_good_months_strict.py

Pipeline 2-en-1 :
  1) Scan qualité micro-structure (BOOK + TRADES) mensuelle
     => microio_scan.csv
  2) Sélection stricte des mois "bons" pour le ML
     => good_months.csv (colonnes: symbol, ym)

- Entrées:
    s3://.../level_1/<SYMBOL>/<YYYY-MM>.parquet
    s3://.../trade/<SYMBOL>/<YYYY-MM>.parquet   (optionnel mais recommandé)
- Sorties:
    s3://.../qa/microio_scan.csv
    s3://.../qa/good_months.csv

Usage typique :
  python build_good_months_strict.py \
    --aws-region ap-northeast-1 \
    --book-root s3://tradebot-config-tokyo/data/level_1 \
    --trade-root s3://tradebot-config-tokyo/data/trade \
    --symbols BTCUSDT ETHUSDT SOLUSDT ... \
    --years 2023 2024 2025 \
    --out-scan-csv s3://tradebot-config-tokyo/qa/microio_scan.csv \
    --out-good-months s3://tradebot-config-tokyo/qa/good_months.csv
"""

from __future__ import annotations
import argparse
import itertools
from typing import Optional, List

import numpy as np
import pandas as pd
import pyarrow.fs as pafs
import pyarrow.dataset as ds

BOOK_LEVELS = 15

# ────────────────────── Seuils par symbole ──────────────────────
# _default : garde les valeurs "globales" (les args --strict-*)
# BTCUSDT / ETHUSDT : overrides symbol-spécifiques sur les seuils stricts.
#
# ⚠️ Si tu veux ajuster les valeurs issues de microio_scan_annotated.csv,
#    il suffit de modifier les nombres ci-dessous.
STRICT_THRESHOLDS = {
    "_default": {
        # ces champs servent juste de fallback; on utilisera
        # args.min_rows_book / args.min_rows_trades / args.strict-* si absent
        "min_rows_book": None,
        "min_rows_trades": None,
        "strict_gap_max": None,
        "strict_crossed_max": None,
        "strict_seq_decr_max": None,
        "strict_lat_neg_max": None,
        "strict_lat_ms99_max": None,
    },
    "BTCUSDT": {
        "min_rows_book": None,         # -> utilise args.min_rows_book
        "min_rows_trades": None,       # -> utilise args.min_rows_trades
        "strict_gap_max": 0.20,        # plus tolérant en présence de gaps
        "strict_crossed_max": 9000,    # + large que défaut 5000
        "strict_seq_decr_max": 100,    # contrôle fort sur seq qui recule
        "strict_lat_neg_max": 600,     # un peu de latitude sur lat_neg
        "strict_lat_ms99_max": None,   # garde valeur globale (20000)
    },
    "ETHUSDT": {
        "min_rows_book": None,
        "min_rows_trades": None,
        "strict_gap_max": 0.20,
        "strict_crossed_max": 9000,
        "strict_seq_decr_max": 100,
        "strict_lat_neg_max": 600,
        "strict_lat_ms99_max": None,
    },
}

# ─────────────────────────── Helpers S3 ────────────────────────────

def _s3fs(region: Optional[str]) -> pafs.S3FileSystem:
    return pafs.S3FileSystem(region=region) if region else pafs.S3FileSystem()

def _s3_object(path: str) -> str:
    return path[len("s3://"):] if path.startswith("s3://") else path

def _exists(fs: pafs.S3FileSystem, s3uri: str) -> bool:
    obj = _s3_object(s3uri)
    try:
        return fs.get_file_info([obj])[0].type != pafs.FileType.NotFound
    except Exception:
        return False

def _fmt(x):
    try:
        return pd.Timestamp(x).isoformat()
    except Exception:
        return str(x)

def _read_parquet(fs: pafs.S3FileSystem, s3uri: str, cols: Optional[List[str]] = None) -> pd.DataFrame:
    path = _s3_object(s3uri)
    dataset = ds.dataset(path, filesystem=fs, format="parquet")
    tbl = dataset.to_table(columns=cols)
    return tbl.to_pandas(types_mapper=pd.ArrowDtype)

# ────────────────────── Scan micro-IO par mois ──────────────────────

def _one_month_probe(fs: pafs.S3FileSystem,
                     symbol: str,
                     ym: str,
                     book_root: str,
                     trade_root: Optional[str],
                     batch_size: int = 500_000,
                     sample_max: int = 500_000) -> dict:
    """Analyse (symbol, YYYY-MM) en mode streaming pour limiter la RAM."""
    import random  # noqa

    y, m = map(int, ym.split("-"))
    book = f"{book_root.rstrip('/')}/{symbol}/{y}-{m:02d}.parquet"
    tr   = f"{trade_root.rstrip('/')}/{symbol}/{y}-{m:02d}.parquet" if trade_root else None

    out = {
        "symbol":symbol, "ym":ym,
        "has_book":False, "has_trades":False,
        "rows_book":0, "rows_trades":0,
        "ts_min":None, "ts_max":None,
        "rt_min":None, "rt_max":None,
        "gap_ratio_1s":np.nan, "crossed_count":np.nan,
        "seq_decreases":np.nan,
        "spread_p10":np.nan,"spread_p50":np.nan,"spread_p90":np.nan,
        "lat_ms_p50":np.nan,"lat_ms_p90":np.nan,"lat_ms_p99":np.nan,"lat_neg":np.nan,
        "error_book":None,
    }

    # BOOK obligatoire
    if not _exists(fs, book):
        return out

    out["has_book"] = True
    cols = ["timestamp","received_time","seq","bid_0_price","ask_0_price","bid_0_size","ask_0_size"]

    path = _s3_object(book)
    try:
        dataset = ds.dataset(path, filesystem=fs, format="parquet")
        scanner = dataset.scanner(columns=cols, batch_size=batch_size)
    except Exception as e:
        out["error_book"] = f"{type(e).__name__}: {e}"
        return out

    # Accumulateurs streaming
    rows_book = 0
    ts_min = None
    ts_max = None
    rt_min = None
    rt_max = None
    lat_neg = 0

    # Pour gap_ratio_1s : set de secondes où on a AU MOINS un tick
    seconds_seen = set()

    # Pour seq_decreases : garder le dernier seq du batch précédent
    last_seq_global = None
    seq_decreases = 0

    # Pour crossed_count
    crossed_count = 0

    # Samples pour spreads & latence (pour quantiles)
    spread_samples = []
    lat_samples = []

    def _reservoir_append(sample_list, values, k):
        """Ajoute values dans sample_list avec un reservoir sampling basique."""
        if not len(values):
            return
        total = len(sample_list) + len(values)
        if total <= k:
            sample_list.extend(values)
            return
        # Si on dépasse, on concat puis on sous-échantillonne
        tmp = np.concatenate([np.array(sample_list), values])
        if len(tmp) > k:
            idx = np.random.choice(len(tmp), size=k, replace=False)
            sample_list[:] = tmp[idx].tolist()
        else:
            sample_list[:] = tmp.tolist()

    # ── Lecture par batches ──
    try:
        for batch in scanner.to_batches():
            df = batch.to_pandas(types_mapper=pd.ArrowDtype)

            if df.empty:
                continue

            rows_book += len(df)

            # timestamp / received_time
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            if "received_time" in df.columns:
                df["received_time"] = pd.to_datetime(df["received_time"], utc=True, errors="coerce")

            df = df.dropna(subset=["timestamp"])
            if df.empty:
                continue

            # ts_min / ts_max
            ts_min_batch = df["timestamp"].min()
            ts_max_batch = df["timestamp"].max()
            ts_min = ts_min_batch if ts_min is None else min(ts_min, ts_min_batch)
            ts_max = ts_max_batch if ts_max is None else max(ts_max, ts_max_batch)

            # seconds_seen pour gap_ratio_1s
            ts_np = df["timestamp"].to_numpy(dtype="datetime64[ns]")
            secs = (ts_np.astype("int64") // 10**9)  # int64 seconds
            seconds_seen.update(secs.tolist())

            # Latence BOOK
            if "received_time" in df.columns:
                ok = df["timestamp"].notna() & df["received_time"].notna()
                if ok.any():
                    dms = (df.loc[ok,"received_time"] - df.loc[ok,"timestamp"]).dt.total_seconds() * 1000.0
                    dms = dms.replace([np.inf, -np.inf], np.nan).dropna()
                    if not dms.empty:
                        lat_neg += int((dms < 0).sum())
                        _reservoir_append(lat_samples, dms.values.astype(float), sample_max)
                        rt_min_batch = df.loc[ok,"received_time"].min()
                        rt_max_batch = df.loc[ok,"received_time"].max()
                        rt_min = rt_min_batch if rt_min is None else min(rt_min, rt_min_batch)
                        rt_max = rt_max_batch if rt_max is None else max(rt_max, rt_max_batch)

            # BOOK croisé (bid >= ask)
            chk = df[["bid_0_price","ask_0_price"]].dropna()
            if not chk.empty:
                crossed_count += int((chk["bid_0_price"] >= chk["ask_0_price"]).sum())

            # Seq monotone
            if "seq" in df.columns:
                seq = pd.to_numeric(df["seq"], errors="coerce").dropna()
                if not seq.empty:
                    # intra-batch
                    diff = seq.diff()
                    seq_decreases += int((diff < 0).fillna(False).sum())
                    # inter-batch : premier vs dernier du batch précédent
                    first = seq.iloc[0]
                    last = seq.iloc[-1]
                    if last_seq_global is not None and first < last_seq_global:
                        seq_decreases += 1
                    last_seq_global = last

            # Spread (bps) — on échantillonne
            sp = ((df["ask_0_price"] - df["bid_0_price"]) /
                  ((df["ask_0_price"] + df["bid_0_price"])/2.0))*1e4
            sp = sp.replace([np.inf,-np.inf], np.nan).dropna()
            if not sp.empty:
                _reservoir_append(spread_samples, sp.values.astype(float), sample_max)

    except Exception as e:
        out["error_book"] = f"{type(e).__name__}: {e}"
        return out

    if rows_book == 0 or ts_min is None or ts_max is None:
        # Rien d'exploitable
        out["rows_book"] = rows_book
        return out

    # Remplir out à partir des accumulations
    out["rows_book"] = rows_book
    out["ts_min"], out["ts_max"] = _fmt(ts_min), _fmt(ts_max)
    if rt_min is not None and rt_max is not None:
        out["rt_min"], out["rt_max"] = _fmt(rt_min), _fmt(rt_max)

    out["lat_neg"] = int(lat_neg)

    # gap_ratio_1s : seconds manquantes sur l'intervalle [ts_min, ts_max]
    total_seconds = int((ts_max - ts_min).total_seconds()) + 1
    if total_seconds > 0:
        seen = len(seconds_seen)
        gap_ratio = max(0.0, 1.0 - float(seen) / float(total_seconds))
        out["gap_ratio_1s"] = float(gap_ratio)
    else:
        out["gap_ratio_1s"] = np.nan

    # crossed_count / seq_decreases
    out["crossed_count"] = int(crossed_count)
    out["seq_decreases"] = int(seq_decreases)

    # Quantiles spread
    if spread_samples:
        arr = np.array(spread_samples, dtype=float)
        p = np.nanpercentile(arr, [10,50,90])
        out["spread_p10"], out["spread_p50"], out["spread_p90"] = float(p[0]), float(p[1]), float(p[2])

    # Quantiles latence
    if lat_samples:
        arr = np.array(lat_samples, dtype=float)
        p = np.nanpercentile(arr, [50,90,99])
        out["lat_ms_p50"], out["lat_ms_p90"], out["lat_ms_p99"] = float(p[0]), float(p[1]), float(p[2])

    # ── TRADES (inchangé, petit impact mémoire) ──
    if tr and _exists(fs, tr):
        out["has_trades"] = True
        try:
            trv = _read_parquet(fs, tr, cols=["timestamp","received_time","price","qty"])
            out["rows_trades"] = len(trv)
        except Exception:
            pass

    return out

# ────────────────────────── Étape 1 : SCAN ─────────────────────────

def run_scan(args) -> pd.DataFrame:
    fs = _s3fs(args.aws_region)

    months = [f"{y}-{m:02d}" for y in args.years for m in range(1, 13)]
    rows = []
    for sym, ym in itertools.product(args.symbols, months):
        row = _one_month_probe(fs, sym, ym, args.book_root, args.trade_root)
        rows.append(row)
        print(f"… scan {sym} {ym} done (rows_book={row['rows_book']} rows_trades={row['rows_trades']})", flush=True)

    df = pd.DataFrame(rows)

    # Flags "relaxés" (diagnostic, pas le filtrage strict)
    df["flag_gap_hi"]   = df["gap_ratio_1s"].fillna(1.0) > args.gap_max
    df["flag_crossing"] = df["crossed_count"].fillna(0).astype(float) > float(args.crossed_max)
    df["flag_seq_back"] = df["seq_decreases"].fillna(0).astype(float) > float(args.seq_decr_max)
    df["flag_lat_neg"]  = df["lat_neg"].fillna(0).astype(float) > float(args.lat_neg_max)

    # Écrit microio_scan.csv (toujours)
    fs = _s3fs(args.aws_region)
    out_path = _s3_object(args.out_scan_csv)
    with fs.open_output_stream(out_path) as out:
        out.write(df.to_csv(index=False).encode("utf-8"))
    print(f"✅ microio_scan.csv écrit vers {args.out_scan_csv}")

    return df

# ─────────────────────── Étape 2 : GOOD_MONTHS ─────────────────────

def _symbol_thresholds(sym: str, args):
    """
    Retourne le set de seuils stricts pour un symbole donné,
    en combinant les valeurs globales (args) et les overrides STRICT_THRESHOLDS.
    """
    base = {
        "min_rows_book": args.min_rows_book,
        "min_rows_trades": args.min_rows_trades,
        "strict_gap_max": args.strict_gap_max,
        "strict_crossed_max": args.strict_crossed_max,
        "strict_seq_decr_max": args.strict_seq_decr_max,
        "strict_lat_neg_max": args.strict_lat_neg_max,
        "strict_lat_ms99_max": args.strict_lat_ms99_max,
    }
    spec = STRICT_THRESHOLDS.get(sym, STRICT_THRESHOLDS.get("_default", {}))
    out = base.copy()
    for k, v in (spec or {}).items():
        if v is not None:
            out[k] = v
    return out

def build_good_months(df: pd.DataFrame, args) -> pd.DataFrame:
    """
    Applique des critères STRICTS pour retenir (symbol, ym) :

    - has_book == True
    - (si trade_root fourni) has_trades == True
    - rows_book   >= min_rows_book (global ou override par symbole)
    - rows_trades >= min_rows_trades (si trade_root, global ou override)
    - gap_ratio_1s <= strict_gap_max (global ou override)
    - crossed_count <= strict_crossed_max
    - seq_decreases <= strict_seq_decr_max
    - lat_neg <= strict_lat_neg_max
    - lat_ms_p99 <= strict_lat_ms99_max
    - pas d'erreur_book
    """
    df = df.copy()

    # On prépare un vecteur de strict_ok à False, puis on applique ligne par ligne
    strict_ok = []
    for idx, row in df.iterrows():
        sym = str(row.get("symbol", ""))
        th = _symbol_thresholds(sym, args)

        ok = True

        # BOOK obligatoire + pas d'erreur
        ok = ok and bool(row.get("has_book", False))
        ok = ok and pd.isna(row.get("error_book", None))

        # TRADES requis uniquement si on a fourni un trade_root
        if args.trade_root:
            ok = ok and bool(row.get("has_trades", False))

        # Min rows
        rows_book = float(row.get("rows_book", 0) or 0)
        rows_trades = float(row.get("rows_trades", 0) or 0)
        if rows_book < th["min_rows_book"]:
            ok = False
        if args.trade_root and rows_trades < th["min_rows_trades"]:
            ok = False

        # Gaps / crossing / seq / latence
        gap_ratio_1s = float(row.get("gap_ratio_1s", 1.0) or 1.0)
        crossed_count = float(row.get("crossed_count", 0) or 0)
        seq_decreases = float(row.get("seq_decreases", 0) or 0)
        lat_neg = float(row.get("lat_neg", 0) or 0)
        lat_ms_p99 = float(row.get("lat_ms_p99", th["strict_lat_ms99_max"] + 1))

        if gap_ratio_1s > th["strict_gap_max"]:
            ok = False
        if crossed_count > th["strict_crossed_max"]:
            ok = False
        if seq_decreases > th["strict_seq_decr_max"]:
            ok = False
        if lat_neg > th["strict_lat_neg_max"]:
            ok = False
        if lat_ms_p99 > th["strict_lat_ms99_max"]:
            ok = False

        strict_ok.append(ok)

    df["strict_ok"] = strict_ok

    kept = df[df["strict_ok"]].copy()
    dropped = df[~df["strict_ok"]].copy()

    print("\nRésumé filtrage strict :")
    print(f"  total mois analysés : {len(df)}")
    print(f"  mois retenus (strict_ok) : {len(kept)}")
    print(f"  mois rejetés : {len(dropped)}")

    # Table symbol/ym (good_months)
    gm = (
        kept[["symbol","ym"]]
        .dropna()
        .drop_duplicates()
        .sort_values(["symbol","ym"])
        .reset_index(drop=True)
    )

    print(f"\ngood_months distincts : {len(gm)} lignes (symbol, ym)")

    # Écrit good_months.csv
    fs = _s3fs(args.aws_region)
    out_path = _s3_object(args.out_good_months)
    with fs.open_output_stream(out_path) as out:
        out.write(gm.to_csv(index=False).encode("utf-8"))
    print(f"✅ good_months.csv écrit vers {args.out_good_months}")

    # Version annotée (strict_ok) si demandé
    if args.out_scan_annotated:
        ann_path = _s3_object(args.out_scan_annotated)
        with fs.open_output_stream(ann_path) as out:
            out.write(df.to_csv(index=False).encode("utf-8"))
        print(f"✅ microio_scan_annotated.csv écrit vers {args.out_scan_annotated}")

    return gm

# ───────────────────────────── CLI ─────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Pipeline strict : microIO scan + good_months.csv pour Stage2"
    )
    p.add_argument("--aws-region", default="ap-northeast-1")

    p.add_argument("--book-root", required=True,
                   help="Racine BOOK15, ex: s3://.../data/level_1")
    p.add_argument("--trade-root", default=None,
                   help="Racine TRADES, ex: s3://.../data/trade (optionnel mais recommandé)")

    p.add_argument("--symbols", nargs="+", required=True,
                   help="Liste des symboles, ex: BTCUSDT ETHUSDT SOLUSDT ...")
    p.add_argument("--years", nargs="+", type=int, required=True,
                   help="Liste des années, ex: 2023 2024 2025")

    p.add_argument("--out-scan-csv", required=True,
                   help="Chemin S3 pour microio_scan.csv")
    p.add_argument("--out-good-months", required=True,
                   help="Chemin S3 pour good_months.csv (symbol, ym)")
    p.add_argument("--out-scan-annotated", default=None,
                   help="(Optionnel) S3 pour microio_scan_annotated.csv avec strict_ok")

    # ── Seuils "relaxés" pour flags (diagnostic uniquement) ──
    p.add_argument("--gap-max", type=float, default=0.80,
                   help="Seuil gap_ratio_1s pour flag_gap_hi (diagnostic, défaut: 0.80)")
    p.add_argument("--crossed-max", type=int, default=1_000,
                   help="Seuil crossed_count pour flag_crossing (diagnostic, défaut: 1000)")
    p.add_argument("--seq-decr-max", type=int, default=10_000,
                   help="Seuil seq_decreases pour flag_seq_back (diagnostic, défaut: 10000)")
    p.add_argument("--lat-neg-max", type=int, default=100,
                   help="Seuil lat_neg pour flag_lat_neg (diagnostic, défaut: 100)")

    # ── Seuils STRICTS (good_months, fallback global) ──
    p.add_argument("--min-rows-book", type=int, default=300000)
    p.add_argument("--min-rows-trades", type=int, default=200000)

    p.add_argument("--strict-gap-max", type=float, default=0.55)
    p.add_argument("--strict-crossed-max", type=int, default=5000)
    p.add_argument("--strict-seq-decr-max", type=int, default=120000)
    p.add_argument("--strict-lat-neg-max", type=int, default=5000)
    p.add_argument("--strict-lat-ms99-max", type=float, default=20000.0)

    return p.parse_args()

# ───────────────────────────── main ────────────────────────────────

def main():
    args = parse_args()

    print("════════ Stage2 — build_good_months_strict.py ════════")
    print(f"AWS region : {args.aws_region}")
    print(f"BOOK root  : {args.book_root}")
    print(f"TRADE root : {args.trade_root}")
    print(f"Symbols    : {args.symbols}")
    print(f"Years      : {args.years}")
    print(f"Out scan   : {args.out_scan_csv}")
    print(f"Out gm     : {args.out_good_months}")
    if args.out_scan_annotated:
        print(f"Out scan+  : {args.out_scan_annotated}")
    print("───────────────────────────────────────────────────────", flush=True)

    # Étape 1 : SCAN
    df_scan = run_scan(args)

    # Étape 2 : GOOD_MONTHS strict (avec thresholds symbol-spécifiques)
    build_good_months(df_scan, args)

    print("Done.")

if __name__ == "__main__":
    main()