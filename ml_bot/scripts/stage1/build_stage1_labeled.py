#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_stage1_labeled.py

Stage1 — RR labeling (trading-aware) sur book L1 bid/ask, RAM-safe (5s only)

- 1 ligne par (symbol, t) sur une base TF (le TF le plus court demandé, typiquement 5m)
- Pour chaque TF:
    y_<tf> (+1 TP_LONG, -1 TP_SHORT, 0 TIME/NOFILL),
    pnl_net_bps_<tf> (TIME/NOFILL => 0),
    tp_bps_<tf>, sl_bps_<tf>, rr_min_<tf>, risk_r_bps_<tf>, ...

Book:
- Path sur bid/ask (L1) resamplé 5s (last + ffill)
- Spread_bps calculé depuis bid/ask

Coûts / conventions:
- Entrée maker (limit FOK): fill check conservateur = "touch" dans une petite fenêtre
- Coût entrée: 0.5 * spread_bps_entry (proxy conservateur)
- Sortie taker: fee_exit_bps fixe (default 6.0)
- TIME et NOFILL => pnl_net_bps = 0 (pas de trade)

RR:
- R_bps = max(risk_mult * ATR_bps, risk_floor_bps)
- TP_bps(tf) = rr_min(tf) * R_bps
- SL_bps(tf) = R_bps
- rr_min(tf) vient de:
    1) --rr-min-map-json (si fourni),
    2) sinon RR_MIN_MAP (constante),
    3) sinon fallback --rr-min.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import fsspec
import sys, gc, time, logging
import platform
import pyarrow.parquet as pq
import pyarrow as pa
import pyarrow.fs as pafs
import hashlib


# ============================================================
# CONFIG
# ============================================================

TF_HORIZONS_SEC: Dict[str, int] = {
    "5m":  300,
    "15m": 900,
    "30m": 1800,
    "1h":  3600,
    "2h":  7200,
    "4h":  14400,
}

SYMBOLS_CANON = [
    "APTUSDT","BCHUSDT","BNBUSDT","BTCUSDT","CRVUSDT",
    "DOGEUSDT","DOTUSDT","ETHUSDT","LINKUSDT","LTCUSDT",
    "OPUSDT","SOLUSDT","SUIUSDT","WLDUSDT","XRPUSDT"
]

# Default RR map (override possible with --rr-min-map-json)
RR_MIN_MAP: Dict[str, float] = {
    #"5m":  1.5,
    "15m": 2.0,
    "30m": 3.0,
    "1h":  3.5,
    "2h":  4.0,
    "4h":  4.5,
}


# ============================================================
# Logging / utils
# ============================================================
def _row_id(symbol: str, t_iso: str, side_key: str = "both") -> str:
    """
    ID stable (déterministe) pour aligner stage2/stage3/stage4 sans dépendre de l'ordre.
    - symbol: ex "BTCUSDT"
    - t_iso: ex "2025-01-01T00:00:00Z" (tu l'as déjà)
    - side_key: doit rester IDENTIQUE pour GO+DIR. Par défaut "both".
    """
    key = f"{symbol}|{t_iso}|{side_key}".encode("utf-8")
    return hashlib.blake2b(key, digest_size=16).hexdigest()  # 32 hex chars

def _log(logger, msg: str):
    logger.info(msg)
    for h in logger.handlers:
        try:
            h.flush()
        except Exception:
            pass
    sys.stdout.flush()
    sys.stderr.flush()

def _setup_logger(verbose: bool = True) -> logging.Logger:
    logger = logging.getLogger("stage1")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    h = logging.StreamHandler(stream=sys.stdout)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    h.setFormatter(fmt)
    logger.addHandler(h)
    if not verbose:
        logger.setLevel(logging.WARNING)
    return logger

def _df_mem_mb(df: pd.DataFrame) -> float:
    try:
        return float(df.memory_usage(deep=True).sum()) / (1024**2)
    except Exception:
        return float("nan")

def _now() -> float:
    return time.perf_counter()

def _symbol_id(sym: str) -> int:
    return SYMBOLS_CANON.index(sym) if sym in SYMBOLS_CANON else -1

def _ensure_utc(ts):
    return pd.to_datetime(ts, utc=True, errors="coerce")

def _to_pandas_freq(tf: str) -> str:
    tf = (tf or "").lower().strip()
    if tf.endswith("m"):
        n = int(tf[:-1])
        return f"{n}min"
    return tf  # "1h","2h","4h"

def _s3_so(region: str | None) -> dict:
    return {"client_kwargs": {"region_name": region}} if region else {}

def _exists(path: str, so: dict) -> bool:
    fs, _, _ = fsspec.get_fs_token_paths(path, storage_options=so)
    return fs.exists(path)

def _pick_base_tf(tfs: List[str]) -> str:
    def tf_to_sec(tf: str) -> int:
        return int(TF_HORIZONS_SEC.get(tf, 10**18))
    tfs_ok = [tf for tf in tfs if tf in TF_HORIZONS_SEC]
    return min(tfs_ok, key=tf_to_sec) if tfs_ok else "5m"

def _downcast_book(df: pd.DataFrame) -> pd.DataFrame:
    for c in ("bid0","ask0","mid","spread_bps"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], downcast="float")
    return df

# ============================
# PATCHES (outside process_symbol_year)
# ============================

def _mfe_mae_and_barriers_R(
    seg_exit: np.ndarray,   # LONG: bid segment, SHORT: ask segment
    entry_px: float,
    R_bps: float,
    side: str,              # "L" or "S"
    k_pos: Tuple[int, ...] = (1, 2, 3, 4, 5),
    k_neg: Tuple[int, ...] = (1, 2, 3),
) -> dict:
    """
    Post-entry (NON-FEATURE) audit stats on an exit-price segment:
      - MFE/MAE in bps and in R
      - flags: hit +kR / -kR
      - first touch: which +/-kR was hit first + step index from fill

    Returns dict keys:
      mfe_bps, mae_bps, mfe_R, mae_R,
      hit_p{k}R, hit_m{k}R,
      first_touch (signed int), first_touch_step
    """
    if (
        seg_exit is None
        or seg_exit.size == 0
        or (not np.isfinite(entry_px))
        or entry_px <= 0
        or (not np.isfinite(R_bps))
        or R_bps <= 0
    ):
        out = {
            "mfe_bps": np.nan, "mae_bps": np.nan,
            "mfe_R": np.nan, "mae_R": np.nan,
            "first_touch": 0, "first_touch_step": -1,
        }
        for k in k_pos:
            out[f"hit_p{k}R"] = 0
        for k in k_neg:
            out[f"hit_m{k}R"] = 0
        return out

    # define favorable/adverse so that "favorable_bps >= 0" always means favorable
    if side == "L":
        favorable_bps = (seg_exit / entry_px - 1.0) * 1e4
        adverse_bps   = (1.0 - seg_exit / entry_px) * 1e4
    else:  # "S"
        favorable_bps = (1.0 - seg_exit / entry_px) * 1e4
        adverse_bps   = (seg_exit / entry_px - 1.0) * 1e4

    # numeric safety
    # numeric safety (avoid NaN => +/-inf that would create fake touches)
    favorable_bps = np.asarray(favorable_bps, dtype=np.float64)
    adverse_bps   = np.asarray(adverse_bps,   dtype=np.float64)

    ok_f = np.isfinite(favorable_bps)
    ok_a = np.isfinite(adverse_bps)

    # If everything is NaN/inf, return safe defaults (no touches)
    if (not ok_f.any()) or (not ok_a.any()):
        out = {
            "mfe_bps": 0.0, "mae_bps": 0.0,
            "mfe_R": 0.0, "mae_R": 0.0,
            "first_touch": 0, "first_touch_step": -1,
        }
        for k in k_pos:
            out[f"hit_p{k}R"] = 0
        for k in k_neg:
            out[f"hit_m{k}R"] = 0
        return out

    # Use only finite values
    f = favorable_bps[ok_f]
    a = adverse_bps[ok_a]

    mfe_bps = float(np.max(f))
    mae_bps = float(np.max(a))

    mfe_R = float(mfe_bps / R_bps)
    mae_R = float(mae_bps / R_bps)

    hits_pos = {k: int(np.any(favorable_bps >= (k * R_bps))) for k in k_pos}
    hits_neg = {k: int(np.any(adverse_bps   >= (k * R_bps))) for k in k_neg}

    # earliest threshold crossing (first touch)
    first_step = 10**12
    first_code = 0

    for k in k_pos:
        idx = np.flatnonzero(favorable_bps >= (k * R_bps))
        if idx.size:
            s = int(idx[0])
            if s < first_step:
                first_step = s
                first_code = +k

    for k in k_neg:
        idx = np.flatnonzero(adverse_bps >= (k * R_bps))
        if idx.size:
            s = int(idx[0])
            if s < first_step:
                first_step = s
                first_code = -k

    if first_step == 10**12:
        first_step = -1
        first_code = 0

    out = {
        "mfe_bps": mfe_bps, "mae_bps": mae_bps,
        "mfe_R": mfe_R, "mae_R": mae_R,
        "first_touch": int(first_code),
        "first_touch_step": int(first_step),
    }
    for k, v in hits_pos.items():
        out[f"hit_p{k}R"] = v
    for k, v in hits_neg.items():
        out[f"hit_m{k}R"] = v
    return out

# ============================================================
# ATR (candles)
# ============================================================

def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    c = df["close"].astype(float)
    pc = c.shift(1)
    tr = pd.concat([(h-l).abs(), (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()


# ============================================================
# Book normalization + read/resample (RAM-safe 5s only)
# ============================================================

def _read_book_l1_resampled_5s(
    book_glob: str,
    so: dict,
    logger: logging.Logger,
    batch_rows: int = 250_000,
    compact_every: int = 200,
) -> pd.DataFrame:
    """
    Fast path:
      - Use pyarrow.fs.S3FileSystem (avoid fsspec file-like for ParquetFile)
      - Read only needed columns (timestamp, bid, ask)
      - Avoid pd.concat on each batch (collect per-file then concat once)
      - Add timing logs to locate stalls
    """
    fs_fsspec, _, paths = fsspec.get_fs_token_paths(book_glob, storage_options=so)
    paths = sorted(paths)
    if not paths:
        _log(logger, f"[WARN] book glob empty: {book_glob} -> SKIP")
        return pd.DataFrame(columns=["bid0", "ask0", "mid", "spread_bps"])

    t0 = time.perf_counter()
    _log(logger, f"book[5s] files={len(paths)} glob={book_glob}")

    # ---------
    # Build a native PyArrow FS for S3 (way faster / less flaky than file-like wrappers)
    # ---------
    region = None
    try:
        region = (so or {}).get("client_kwargs", {}).get("region_name")
    except Exception:
        region = None

    # PyArrow S3FileSystem uses env/instance creds; region helps.
    s3 = pafs.S3FileSystem(region=region) if region else pafs.S3FileSystem()

    # Helper: convert s3://bucket/key -> "bucket/key" for pyarrow.fs
    def _to_pa_path(p: str) -> str:
        p = str(p)
        if p.startswith("s3://"):
            return p[len("s3://"):]
        return p

    out_parts: List[pd.DataFrame] = []
    appended = 0

    for fi, p in enumerate(paths, 1):
        _log(logger, f"book[5s] open file {fi}/{len(paths)}: {p}")
        p_pa = _to_pa_path(p)

        t_file0 = time.perf_counter()
        try:
            # IMPORTANT: pass filesystem + path string, no fs.open()
            pf = pq.ParquetFile(p_pa, filesystem=s3)
        except Exception as e:
            _log(logger, f"[ERROR] ParquetFile failed for {p}: {type(e).__name__}: {e}")
            continue

        _log(logger, f"book[5s] ParquetFile ready rows≈{pf.metadata.num_rows:,} row_groups={pf.num_row_groups} ({time.perf_counter()-t_file0:.2f}s)")

        # Determine columns once from schema (case-insensitive)
        schema_names = [n for n in pf.schema.names]
        cols_lower = {n.lower(): n for n in schema_names}

        if "timestamp" not in cols_lower:
            _log(logger, f"[WARN] missing timestamp in {p} -> skip")
            continue
        ts_col = cols_lower["timestamp"]

        bid_col = ask_col = None
        for b, a in [("bid", "ask"), ("bid_0_price", "ask_0_price")]:
            if b in cols_lower and a in cols_lower:
                bid_col, ask_col = cols_lower[b], cols_lower[a]
                break
        if bid_col is None or ask_col is None:
            _log(logger, f"[WARN] missing bid/ask in {p} -> skip")
            continue

        # Read only needed columns
        read_cols = [ts_col, bid_col, ask_col]

        # Collect resampled chunks for this file, concat once (avoid quadratic concat)
        file_parts: List[pd.DataFrame] = []

        t_batches0 = time.perf_counter()
        first_batch_logged = False

        for bi, batch in enumerate(pf.iter_batches(batch_size=batch_rows, columns=read_cols), 1):
            if (not first_batch_logged):
                _log(logger, f"book[5s] first batch received for file {fi} after {time.perf_counter()-t_batches0:.2f}s")
                first_batch_logged = True

            tbl = pa.Table.from_batches([batch])
            df = tbl.to_pandas(self_destruct=True)

            if df is None or df.empty:
                continue

            # normalize to standard names without rebuilding huge df
            df.rename(columns={ts_col: "timestamp", bid_col: "bid0", ask_col: "ask0"}, inplace=True)

            df["timestamp"] = _ensure_utc(df["timestamp"])
            df = df.dropna(subset=["timestamp"]).sort_values("timestamp")

            # numeric coercion
            df["bid0"] = pd.to_numeric(df["bid0"], errors="coerce")
            df["ask0"] = pd.to_numeric(df["ask0"], errors="coerce")
            df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["bid0", "ask0"])

            # Drop crossed / invalid quotes (ask < bid) BEFORE mid/spread
            bad = (df["bid0"] <= 0) | (df["ask0"] <= 0) | (df["ask0"] < df["bid0"])
            if bad.any():
                # option: log a few counts occasionally (or per file)
                # _log(logger, f"book[5s] dropped crossed/invalid: {int(bad.sum())}/{len(df)}")
                df = df[~bad]
            if df.empty:
                continue

            df.set_index("timestamp", inplace=True)

            # compute mid + spread only after cleaning
            mid = (df["bid0"].astype("float64") + df["ask0"].astype("float64")) / 2.0
            spread_bps = ((df["ask0"].astype("float64") - df["bid0"].astype("float64")) / mid) * 1e4
            df["mid"] = mid
            df["spread_bps"] = spread_bps

            g = df[["bid0", "ask0", "mid", "spread_bps"]].resample("5s").last().ffill()
            if not g.empty:
                for c in ("bid0", "ask0", "mid", "spread_bps"):
                    g[c] = pd.to_numeric(g[c], downcast="float")
                file_parts.append(g)
                appended += 1

            del df, g, tbl
            if (bi % 25) == 0:
                _log(logger, f"book[5s] file {fi} batch {bi} appended={appended} file_parts={len(file_parts)}")
            gc.collect()

        if not file_parts:
            _log(logger, f"book[5s] file {fi} produced no data (elapsed={time.perf_counter()-t_file0:.1f}s)")
            continue

        file_df = pd.concat(file_parts, axis=0)
        file_df = file_df.sort_index()
        file_df = file_df[~file_df.index.duplicated(keep="last")]
        out_parts.append(file_df)

        _log(logger, f"book[5s] file {fi} done rows={len(file_df):,} elapsed={time.perf_counter()-t_file0:.1f}s")
        del file_parts, file_df
        gc.collect()

    if not out_parts:
        return pd.DataFrame(columns=["bid0", "ask0", "mid", "spread_bps"])

    out = pd.concat(out_parts, axis=0)
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]
    _log(logger, f"book[5s] done rows={len(out):,} elapsed={time.perf_counter()-t0:.1f}s mem≈{_df_mem_mb(out):.1f}MB")
    return out

# ============================================================
# RR labeling (bid/ask path) + conservative fill
# ============================================================

def _first_true_idx(mask: np.ndarray) -> int:
    idx = np.flatnonzero(mask)
    return int(idx[0]) if idx.size else -1

def _maker_fill_ok_conservative(
    mid_arr: np.ndarray,
    p0: int,
    fill_w: int,
    entry_bid: float,
    entry_ask: float,
) -> Tuple[bool, bool, int, int]:
    """
    Option 3 (ML-friendly): maker fill proxy via mid-cross.
      - Long fill possible if mid <= entry_bid within [p0, p0+fill_w]
      - Short fill possible if mid >= entry_ask within [p0, p0+fill_w]
    Returns:
      okL, okS, fill_step_L, fill_step_S
    """
    if p0 < 0 or p0 >= mid_arr.size:
        return False, False, -1, -1

    p1 = min(p0 + fill_w, mid_arr.size - 1)
    if p1 <= p0:
        return False, False, -1, -1

    seg_mid = mid_arr[p0:p1 + 1]

    fill_step_L = _first_true_idx(seg_mid <= entry_bid)
    fill_step_S = _first_true_idx(seg_mid >= entry_ask)

    okL = (fill_step_L >= 0)
    okS = (fill_step_S >= 0)
    return okL, okS, int(fill_step_L), int(fill_step_S)

def _side_tp_sl_steps_after_fill(
    seg_bid: np.ndarray,
    seg_ask: np.ndarray,
    entry_long: float,
    entry_short: float,
    tp_bps: float,
    sl_bps: float,
) -> Dict[str, int]:
    """
    Returns first indices (steps) for each event within the provided segment:
      tpL, slL, tpS, slS  (each -1 if not hit)
    Convention:
      LONG exits on bid ; SHORT exits on ask.
    """
    if seg_bid.size < 2 or seg_ask.size < 2:
        return {"tpL": -1, "slL": -1, "tpS": -1, "slS": -1}

    tpL_px = entry_long * (1.0 + tp_bps / 1e4)
    slL_px = entry_long * (1.0 - sl_bps / 1e4)
    tpS_px = entry_short * (1.0 - tp_bps / 1e4)
    slS_px = entry_short * (1.0 + sl_bps / 1e4)

    tpL = _first_true_idx(seg_bid >= tpL_px)
    slL = _first_true_idx(seg_bid <= slL_px)
    tpS = _first_true_idx(seg_ask <= tpS_px)
    slS = _first_true_idx(seg_ask >= slS_px)

    return {"tpL": int(tpL), "slL": int(slL), "tpS": int(tpS), "slS": int(slS)}

# ============================================================
# Core
# ============================================================

# ============================
# NEW VERSION: process_symbol_year (Stage1 v3 audits)
# ============================

def process_symbol_year(
    symbol: str,
    year: int,
    book_root: str,
    bougie_root: str,
    out_root: str,
    tfs: List[str],
    aws_region: str | None,
    atr_mid_proxy: str,
    risk_mult: float,
    rr_min_default: float,
    rr_min_map: Dict[str, float],
    risk_floor_bps: float,
    fee_exit_bps: float,
    fill_window_sec: int,
):
    so = _s3_so(aws_region)
    book_glob = f"{book_root.rstrip('/')}/{symbol}/{year}-*.parquet"

    logger = _setup_logger()
    logger.info(f"START {symbol} {year} | python={platform.python_version()} pandas={pd.__version__}")
    t0 = _now()

    # -------------------------
    # BOOK 5s ONLY (RAM-safe)
    # -------------------------
    logger.info(f"{symbol} {year} BOOK dt=5s only (RR bid/ask)")
    book_5s = _read_book_l1_resampled_5s(book_glob, so, logger=logger)
    if book_5s is None or book_5s.empty:
        logger.warning(f"SKIP {symbol} {year}: no book data for glob={book_glob}")
        return

    book_5s = _downcast_book(book_5s)
    logger.info(f"book_5s rows={len(book_5s):,} mem≈{_df_mem_mb(book_5s):.1f}MB")

    book_mid_1m = book_5s["mid"].resample("1min").last().ffill()

    # -------------------------
    # CANDLES (ATR only)
    # -------------------------
    candle_path = f"{bougie_root.rstrip('/')}/{symbol}/{symbol}-1m-{year}.parquet"
    if not _exists(candle_path, so):
        raise FileNotFoundError(f"Missing candles (ATR): {candle_path}")

    need_cols = ["timestamp", "high", "low", "close"]
    if atr_mid_proxy == "oc2":
        need_cols.append("open")

    candles = pd.read_parquet(candle_path, storage_options=so, columns=need_cols)
    if candles is None or candles.empty:
        logger.warning(f"empty candles: {symbol} {year}")
        return

    candles["timestamp"] = _ensure_utc(candles["timestamp"])
    candles = candles.dropna(subset=["timestamp"]).sort_values("timestamp").set_index("timestamp", drop=True)

    if atr_mid_proxy == "oc2":
        mid_proxy = (candles["open"].astype("float64") + candles["close"].astype("float64")) / 2.0
    else:
        mid_proxy = candles["close"].astype("float64")

    candles["atr"] = _atr(candles, 14).astype("float64")

    if atr_mid_proxy == "bookmid":
        mid_ref = book_mid_1m.reindex(candles.index, method="ffill").fillna(mid_proxy)
    else:
        mid_ref = mid_proxy

    candles["atr_bps"] = (candles["atr"] / mid_ref) * 1e4
    candles = candles.replace([np.inf, -np.inf], np.nan)
    candles.drop(columns=["atr"], inplace=True)
    gc.collect()

    # -------------------------
    # BASE GRID (TF le plus court)
    # -------------------------
    tfs = [str(tf).lower().strip() for tf in (tfs or []) if str(tf).lower().strip() in TF_HORIZONS_SEC]
    if not tfs:
        logger.warning(f"no valid TFs for {symbol} {year}")
        return

    base_tf = _pick_base_tf(tfs)
    base_freq = _to_pandas_freq(base_tf)

    base_book = book_5s[["bid0", "ask0", "mid", "spread_bps"]].resample(base_freq).last().dropna(subset=["bid0", "ask0", "mid"])
    if base_book.empty:
        logger.warning(f"empty base resample for {symbol} {year} (base_tf={base_tf})")
        return

    base_atr = candles[["atr_bps"]].resample(base_freq).last()

    out = pd.DataFrame(index=base_book.index)
    out.index.name = "t"

    out["t"] = out.index.map(lambda x: pd.Timestamp(x).isoformat().replace("+00:00", "Z"))
    out["symbol"] = symbol

    # row_id stable pour alignement cross-stages (GO + DIR)
    out["row_id"] = [_row_id(symbol, t, side_key="both") for t in out["t"].tolist()]

    out["year"] = int(year)
    out["symbol_id"] = int(_symbol_id(symbol))

    out["bid_entry"] = base_book["bid0"].astype("float32")
    out["ask_entry"] = base_book["ask0"].astype("float32")
    out["mid_entry"] = base_book["mid"].astype("float32")
    out["spread_bps_entry"] = base_book["spread_bps"].astype("float32")
    out["atr_bps"] = base_atr["atr_bps"].reindex(out.index).astype("float32")

    out["fee_exit_bps"] = np.float32(fee_exit_bps)
    out["entry_spread_half_bps"] = (0.5 * out["spread_bps_entry"]).astype("float32")

    out["risk_mult"] = np.float32(risk_mult)
    out["risk_floor_bps"] = np.float32(risk_floor_bps)
    out["rr_min_default"] = np.float32(rr_min_default)

    out["fill_mode"] = "conservative"
    out["fill_window_sec"] = np.int16(fill_window_sec)

    # -------------------------
    # PREP arrays + indexer (ffill)
    # -------------------------
    bid_path = book_5s["bid0"].astype("float64")
    ask_path = book_5s["ask0"].astype("float64")
    mid_path = book_5s["mid"].astype("float64")

    bid_arr = bid_path.to_numpy(dtype="float64", copy=False)
    ask_arr = ask_path.to_numpy(dtype="float64", copy=False)
    mid_arr = mid_path.to_numpy(dtype="float64", copy=False)

    pos = bid_path.index.get_indexer(out.index, method="ffill").astype(np.int64, copy=False)
    arr_len = int(bid_arr.size)

    # -------------------------
    # Shared per-row params (independent of TF)
    # -------------------------
    step_sec = 5
    fill_w = max(1, int(fill_window_sec // step_sec))

    bid_entry = out["bid_entry"].astype("float64").to_numpy()
    ask_entry = out["ask_entry"].astype("float64").to_numpy()
    cost = out["entry_spread_half_bps"].astype("float64").to_numpy() + float(fee_exit_bps)

    atr_bps_arr = out["atr_bps"].astype("float64").to_numpy()
    nan_atr = int(np.isnan(atr_bps_arr).sum())
    if nan_atr:
        logger.warning(f"{symbol} {year}: atr_bps NaN count={nan_atr}/{atr_bps_arr.size} ({nan_atr/atr_bps_arr.size:.2%})")

    atr_bps_arr = np.nan_to_num(atr_bps_arr, nan=0.0, posinf=0.0, neginf=0.0)
    R_bps_arr = np.maximum(float(risk_mult) * atr_bps_arr, float(risk_floor_bps)).astype(np.float64)

    n = len(out)

    # -------------------------
    # RR labeling + AUDIT per TF
    # -------------------------
    for tf in tfs:
        horizon_sec = int(TF_HORIZONS_SEC[tf])
        window = int(horizon_sec // step_sec)
        if window < 1:
            window = 1

        rr_tf = float(rr_min_map.get(tf, RR_MIN_MAP.get(tf, rr_min_default)))
        TP_bps_arr = (rr_tf * R_bps_arr).astype(np.float64)
        SL_bps_arr = R_bps_arr  # 1R

        # --- EV threshold (EV=0) ---
        cost_bps_arr = cost
        denom = np.maximum(TP_bps_arr + SL_bps_arr, 1e-12)
        p_thr_ev0 = (SL_bps_arr + cost_bps_arr) / denom
        p_thr_ev0 = np.clip(p_thr_ev0, 0.0, 1.0).astype(np.float32)
        if not np.isfinite(p_thr_ev0).all():
            raise RuntimeError(f"{symbol} {year} tf={tf}: p_thr_ev0 contains NaN/inf")

        q = np.quantile(p_thr_ev0, [0.01, 0.5, 0.99])
        logger.info(f"{symbol} {year} tf={tf} p_thr_ev0 quantiles: {q}")
        out[f"p_thr_ev0_{tf}"] = p_thr_ev0

        logger.info(f"{symbol} {year} tf={tf} RR start (h={horizon_sec}s window={window} step={step_sec}s rr_min={rr_tf})")
        t_tf0 = _now()

        out[f"horizon_sec_{tf}"] = np.int32(horizon_sec)
        out[f"step_sec_{tf}"] = np.int16(step_sec)
        out[f"rr_min_{tf}"] = np.float32(rr_tf)

        out[f"risk_r_bps_{tf}"] = R_bps_arr.astype("float32")
        out[f"tp_bps_{tf}"] = TP_bps_arr.astype("float32")
        out[f"sl_bps_{tf}"] = SL_bps_arr.astype("float32")

        # Labels / outcomes
        y_dir = np.zeros(n, dtype=np.int8)
        y_go = np.zeros(n, dtype=np.int8)

        exit_reason = np.empty(n, dtype=object)
        exit_step = np.full(n, -1, dtype=np.int32)
        pnl_net = np.zeros(n, dtype=np.float32)

        # ============================
        # AUDIT (NON-FEATURE) ARRAYS
        # per-side: L and S
        # ============================
        auditL_mfe_R = np.full(n, np.nan, dtype=np.float32)
        auditL_mae_R = np.full(n, np.nan, dtype=np.float32)
        auditL_first = np.zeros(n, dtype=np.int8)
        auditL_first_step = np.full(n, -1, dtype=np.int32)

        auditS_mfe_R = np.full(n, np.nan, dtype=np.float32)
        auditS_mae_R = np.full(n, np.nan, dtype=np.float32)
        auditS_first = np.zeros(n, dtype=np.int8)
        auditS_first_step = np.full(n, -1, dtype=np.int32)

        # flags +kR / -kR
        k_pos = (1, 2, 3, 4, 5)
        k_neg = (1, 2, 3)

        auditL_hit_p = {k: np.zeros(n, dtype=np.int8) for k in k_pos}
        auditL_hit_m = {k: np.zeros(n, dtype=np.int8) for k in k_neg}
        auditS_hit_p = {k: np.zeros(n, dtype=np.int8) for k in k_pos}
        auditS_hit_m = {k: np.zeros(n, dtype=np.int8) for k in k_neg}

        for i in range(n):
            p0 = int(pos[i])
            if p0 < 0 or p0 >= arr_len:
                exit_reason[i] = "NONE"
                continue

            p1 = p0 + window
            if p1 >= arr_len:
                exit_reason[i] = "NONE"
                continue

            eL = float(bid_entry[i])
            eS = float(ask_entry[i])
            if (not np.isfinite(eL)) or (not np.isfinite(eS)) or eL <= 0 or eS <= 0:
                exit_reason[i] = "NONE"
                continue

            okL, okS, fill_step_L, fill_step_S = _maker_fill_ok_conservative(
                mid_arr=mid_arr,
                p0=p0,
                fill_w=fill_w,
                entry_bid=eL,
                entry_ask=eS,
            )

            okL = okL and (p0 + fill_step_L < p1)
            okS = okS and (p0 + fill_step_S < p1)

            if (not okL) and (not okS):
                exit_reason[i] = "NOFILL"
                exit_step[i] = -1
                y_go[i] = 0
                y_dir[i] = 0
                pnl_net[i] = 0.0
                continue

            INF = 10**12

            # -------------------------
            # LONG side (post-fill segment)
            # -------------------------
            if okL:
                p0L = p0 + fill_step_L
                if p0L < p1:
                    seg_bid_L = bid_arr[p0L:p1 + 1]
                    seg_ask_L = ask_arr[p0L:p1 + 1]

                    evL = _side_tp_sl_steps_after_fill(
                        seg_bid=seg_bid_L,
                        seg_ask=seg_ask_L,
                        entry_long=eL,
                        entry_short=eS,
                        tp_bps=float(TP_bps_arr[i]),
                        sl_bps=float(SL_bps_arr[i]),
                    )
                    tpL = evL["tpL"]
                    slL = evL["slL"]

                    # AUDIT: use exit-price path (LONG exits on bid)
                    dL = _mfe_mae_and_barriers_R(
                        seg_exit=seg_bid_L,
                        entry_px=eL,
                        R_bps=float(R_bps_arr[i]),
                        side="L",
                        k_pos=k_pos,
                        k_neg=k_neg,
                    )
                    auditL_mfe_R[i] = np.float32(dL["mfe_R"])
                    auditL_mae_R[i] = np.float32(dL["mae_R"])
                    auditL_first[i] = np.int8(dL["first_touch"])
                    fts = int(dL["first_touch_step"])
                    auditL_first_step[i] = np.int32((fill_step_L + fts) if (fts >= 0 and fill_step_L >= 0) else -1)
                    for k in k_pos:
                        auditL_hit_p[k][i] = np.int8(dL[f"hit_p{k}R"])
                    for k in k_neg:
                        auditL_hit_m[k][i] = np.int8(dL[f"hit_m{k}R"])
                else:
                    tpL = -1
                    slL = -1
            else:
                tpL = -1
                slL = -1

            # -------------------------
            # SHORT side (post-fill segment)
            # -------------------------
            if okS:
                p0S = p0 + fill_step_S
                if p0S < p1:
                    seg_bid_S = bid_arr[p0S:p1 + 1]
                    seg_ask_S = ask_arr[p0S:p1 + 1]

                    evS = _side_tp_sl_steps_after_fill(
                        seg_bid=seg_bid_S,
                        seg_ask=seg_ask_S,
                        entry_long=eL,
                        entry_short=eS,
                        tp_bps=float(TP_bps_arr[i]),
                        sl_bps=float(SL_bps_arr[i]),
                    )
                    tpS = evS["tpS"]
                    slS = evS["slS"]

                    # AUDIT: use exit-price path (SHORT exits on ask)
                    dS = _mfe_mae_and_barriers_R(
                        seg_exit=seg_ask_S,
                        entry_px=eS,
                        R_bps=float(R_bps_arr[i]),
                        side="S",
                        k_pos=k_pos,
                        k_neg=k_neg,
                    )
                    auditS_mfe_R[i] = np.float32(dS["mfe_R"])
                    auditS_mae_R[i] = np.float32(dS["mae_R"])
                    auditS_first[i] = np.int8(dS["first_touch"])
                    fts = int(dS["first_touch_step"])
                    auditS_first_step[i] = np.int32((fill_step_S + fts) if (fts >= 0 and fill_step_S >= 0) else -1)
                    for k in k_pos:
                        auditS_hit_p[k][i] = np.int8(dS[f"hit_p{k}R"])
                    for k in k_neg:
                        auditS_hit_m[k][i] = np.int8(dS[f"hit_m{k}R"])
                else:
                    tpS = -1
                    slS = -1
            else:
                tpS = -1
                slS = -1

            # Decide if each side "wins" (TP before SL)
            def wins(tp_idx: int, sl_idx: int) -> bool:
                if tp_idx < 0:
                    return False
                if sl_idx < 0:
                    return True
                return tp_idx < sl_idx

            long_wins = okL and wins(tpL, slL)
            short_wins = okS and wins(tpS, slS)

            t_tp_long = (fill_step_L + tpL) if (long_wins and tpL >= 0 and fill_step_L >= 0) else INF
            t_tp_short = (fill_step_S + tpS) if (short_wins and tpS >= 0 and fill_step_S >= 0) else INF

            if long_wins or short_wins:
                y_go[i] = 1
                if t_tp_long <= t_tp_short:
                    y_dir[i] = 1
                    exit_reason[i] = "TP_LONG"
                    exit_step[i] = int(t_tp_long)
                else:
                    y_dir[i] = -1
                    exit_reason[i] = "TP_SHORT"
                    exit_step[i] = int(t_tp_short)

                pnl_net[i] = np.float32(float(TP_bps_arr[i]) - cost[i])
                continue

            # Otherwise SL (if any) else TIME
            t_sl_long = (fill_step_L + slL) if (okL and slL >= 0 and fill_step_L >= 0) else INF
            t_sl_short = (fill_step_S + slS) if (okS and slS >= 0 and fill_step_S >= 0) else INF
            t_sl = min(t_sl_long, t_sl_short)

            y_go[i] = 0
            y_dir[i] = 0

            if t_sl < INF:
                if t_sl_long <= t_sl_short:
                    exit_reason[i] = "SL_LONG"
                    exit_step[i] = int(t_sl_long)
                else:
                    exit_reason[i] = "SL_SHORT"
                    exit_step[i] = int(t_sl_short)

                pnl_net[i] = np.float32(-float(SL_bps_arr[i]) - cost[i])
            else:
                exit_reason[i] = "TIME"
                exit_step[i] = -1
                pnl_net[i] = 0.0

        # Backward-compatible labels
        out[f"y_{tf}"] = y_dir
        out[f"y_dir_{tf}"] = y_dir
        out[f"y_go_{tf}"] = y_go

        out[f"exit_reason_{tf}"] = exit_reason
        out[f"exit_step_{tf}"] = exit_step
        out[f"pnl_net_bps_{tf}"] = pnl_net

        # ============================
        # WRITE AUDIT (NON-FEATURE) COLUMNS
        # ============================
        out[f"auditL_mfe_R_{tf}"] = auditL_mfe_R
        out[f"auditL_mae_R_{tf}"] = auditL_mae_R
        out[f"auditL_first_touch_{tf}"] = auditL_first
        out[f"auditL_first_touch_step_{tf}"] = auditL_first_step

        out[f"auditS_mfe_R_{tf}"] = auditS_mfe_R
        out[f"auditS_mae_R_{tf}"] = auditS_mae_R
        out[f"auditS_first_touch_{tf}"] = auditS_first
        out[f"auditS_first_touch_step_{tf}"] = auditS_first_step

        for k in k_pos:
            out[f"auditL_hit_p{k}R_{tf}"] = auditL_hit_p[k]
            out[f"auditS_hit_p{k}R_{tf}"] = auditS_hit_p[k]
        for k in k_neg:
            out[f"auditL_hit_m{k}R_{tf}"] = auditL_hit_m[k]
            out[f"auditS_hit_m{k}R_{tf}"] = auditS_hit_m[k]

        logger.info(f"{symbol} {year} tf={tf} RR+AUDIT done in {_now()-t_tf0:.2f}s")
        gc.collect()

    # -------------------------
    # WRITE CSV
    # -------------------------
    base_cols = [
        "row_id", "t", "symbol", "year", "symbol_id",
        "bid_entry", "ask_entry", "mid_entry",
        "spread_bps_entry", "atr_bps",
        "fee_exit_bps", "entry_spread_half_bps",
        "risk_mult", "risk_floor_bps", "rr_min_default",
        "fill_mode", "fill_window_sec",
    ]

    tf_cols: List[str] = []
    audit_cols: List[str] = []
    for tf in tfs:
        tf_cols += [
            f"horizon_sec_{tf}",
            f"step_sec_{tf}",
            f"rr_min_{tf}",
            f"risk_r_bps_{tf}",
            f"tp_bps_{tf}",
            f"sl_bps_{tf}",
            f"p_thr_ev0_{tf}",
            f"y_{tf}",
            f"y_dir_{tf}",
            f"y_go_{tf}",
            f"exit_reason_{tf}",
            f"exit_step_{tf}",
            f"pnl_net_bps_{tf}",
        ]

        # AUDIT (NON-FEATURE)
        audit_cols += [
            f"auditL_mfe_R_{tf}", f"auditL_mae_R_{tf}",
            f"auditL_first_touch_{tf}", f"auditL_first_touch_step_{tf}",
            f"auditS_mfe_R_{tf}", f"auditS_mae_R_{tf}",
            f"auditS_first_touch_{tf}", f"auditS_first_touch_step_{tf}",
        ]
        for k in (1, 2, 3, 4, 5):
            audit_cols += [f"auditL_hit_p{k}R_{tf}", f"auditS_hit_p{k}R_{tf}"]
        for k in (1, 2, 3):
            audit_cols += [f"auditL_hit_m{k}R_{tf}", f"auditS_hit_m{k}R_{tf}"]

    out_df = out.reset_index(drop=True)[base_cols + tf_cols + audit_cols]

    # free big buffers
    del base_book, base_atr, candles, book_mid_1m, book_5s, bid_path, ask_path, mid_path, bid_arr, ask_arr, mid_arr, pos
    gc.collect()

    out_path = f"{out_root.rstrip('/')}/{symbol}/{year}_signals.csv"
    if out_path.startswith("s3://"):
        fs = fsspec.filesystem("s3", **so)
        parent = os.path.dirname(out_path)
        if not fs.exists(parent):
            fs.mkdirs(parent, exist_ok=True)
        with fsspec.open(out_path, "wb", **so) as f:
            out_df.to_csv(f, index=False)
    else:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(out_path, index=False)

    logger.info(f"DONE {symbol} {year} rows={len(out_df):,} total_time={_now()-t0:.1f}s")
    print(f"[OK] {symbol} {year} → {len(out_df)} rows (base_tf={base_tf}, RR bid/ask, conservative fill) → {out_path}")

# ============================================================
# CLI
# ============================================================

def _parse_rr_min_map_json(s: str | None) -> Dict[str, float]:
    if not s:
        return {}
    obj = json.loads(s)
    if not isinstance(obj, dict):
        raise ValueError("--rr-min-map-json must be a JSON object like {'5m':1.5,'15m':2.0}")
    out: Dict[str, float] = {}
    for k, v in obj.items():
        tf = str(k).lower().strip()
        if tf not in TF_HORIZONS_SEC:
            continue
        out[tf] = float(v)
    return out

def main():
    ap = argparse.ArgumentParser("Stage1 RR labeling (book L1 bid/ask + candles ATR, multi-label per timestamp)")
    ap.add_argument("--book-root", default="s3://tradebot-config-tokyo/data/book")
    ap.add_argument("--bougie-root", default="s3://tradebot-config-tokyo/data/bougie")
    ap.add_argument("--out-root", default="s3://tradebot-config-tokyo/data/new_stage1")
    ap.add_argument("--symbols", nargs="+", default=SYMBOLS_CANON)
    ap.add_argument("--years", nargs="+", type=int, required=True)
    ap.add_argument("--tfs", nargs="+", default=list(TF_HORIZONS_SEC.keys()))
    ap.add_argument("--aws-region", default="ap-northeast-1")

    ap.add_argument("--risk-mult", type=float, default=1.0, help="R = risk_mult * ATR_bps (before floor)")
    ap.add_argument("--rr-min", type=float, default=3.0, help="Fallback rr_min if TF not in map")
    ap.add_argument("--rr-min-map-json", type=str, default=None,
                    help="Override rr_min per TF. Example: '{\"5m\":1.5,\"15m\":2.0,\"1h\":3.0}'")
    ap.add_argument("--risk-floor-bps", type=float, default=8.0, help="min R in bps")

    ap.add_argument("--fee-exit-bps", type=float, default=6.0, help="taker fee at exit (bps)")
    ap.add_argument("--fill-window-sec", type=int, default=10, help="conservative maker fill window in seconds")

    ap.add_argument("--atr-mid-proxy", choices=["close", "oc2", "bookmid"], default="bookmid",
                    help="Prix ref pour normaliser ATR en bps: bookmid (reco), close, oc2=(open+close)/2")
    args = ap.parse_args()

    tfs = [str(tf).lower().strip() for tf in (args.tfs or [])]
    rr_min_map = _parse_rr_min_map_json(args.rr_min_map_json)

    for sym in args.symbols:
        for y in args.years:
            process_symbol_year(
                sym, y,
                book_root=args.book_root,
                bougie_root=args.bougie_root,
                out_root=args.out_root,
                tfs=tfs,
                aws_region=args.aws_region,
                atr_mid_proxy=str(args.atr_mid_proxy),
                risk_mult=float(args.risk_mult),
                rr_min_default=float(args.rr_min),
                rr_min_map=rr_min_map,
                risk_floor_bps=float(args.risk_floor_bps),
                fee_exit_bps=float(args.fee_exit_bps),
                fill_window_sec=int(args.fill_window_sec),
            )

if __name__ == "__main__":
    main()
