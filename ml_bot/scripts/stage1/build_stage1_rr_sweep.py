#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_stage1_rr_sweep.py

Stage1 — RR labeling (trading-aware) sur book L1 bid/ask, RAM-safe (5s only)

- 1 ligne par (symbol, t) sur une base TF (le TF le plus court demandé, typiquement 15m ici)
- Pour chaque TF:
    y_<tf> (+1 TP_LONG, -1 TP_SHORT, 0 TIME/NOFILL),
    pnl_net_bps_<tf> (TIME/NOFILL => 0),
    tp_bps_<tf>, sl_bps_<tf>, rr_min_<tf>, risk_r_bps_<tf>, etc.

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
# (ici volontairement sans 5m)
RR_MIN_MAP: Dict[str, float] = {
    "15m": 2.0,
    "30m": 3.0,
    "1h":  3.5,
    "2h":  4.0,
    "4h":  4.5,
}


# ============================================================
# Logging / utils
# ============================================================

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
    return tf

def _s3_so(region: str | None) -> dict:
    return {"client_kwargs": {"region_name": region}} if region else {}

def _exists(path: str, so: dict) -> bool:
    fs, _, _ = fsspec.get_fs_token_paths(path, storage_options=so)
    return fs.exists(path)

def _pick_base_tf(tfs: List[str]) -> str:
    def tf_to_sec(tf: str) -> int:
        return int(TF_HORIZONS_SEC.get(tf, 10**18))
    tfs_ok = [tf for tf in tfs if tf in TF_HORIZONS_SEC]
    return min(tfs_ok, key=tf_to_sec) if tfs_ok else "15m"

def _downcast_book(df: pd.DataFrame) -> pd.DataFrame:
    for c in ("bid0","ask0","mid","spread_bps"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], downcast="float")
    return df


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

def _normalize_book_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower(): c for c in df.columns}
    if "timestamp" not in cols:
        raise ValueError("Book parquet: missing 'timestamp' column")
    ts_col = cols["timestamp"]

    bid_col = ask_col = None
    for b, a in [("bid", "ask"), ("bid_0_price", "ask_0_price")]:
        if b in cols and a in cols:
            bid_col, ask_col = cols[b], cols[a]
            break
    if bid_col is None or ask_col is None:
        raise ValueError("Book parquet: need bid/ask (bid_0_price/ask_0_price)")

    bid0 = df[bid_col].astype("float64")
    ask0 = df[ask_col].astype("float64")

    # filtre minimal anti-corruption
    m = np.isfinite(bid0) & np.isfinite(ask0) & (bid0 > 0) & (ask0 > 0) & (ask0 >= bid0)
    bid0 = bid0.where(m)
    ask0 = ask0.where(m)

    mid = (bid0 + ask0) / 2.0
    spread_bps = ((ask0 - bid0) / mid) * 1e4

    return pd.DataFrame({
        "timestamp": df[ts_col],
        "bid0": bid0,
        "ask0": ask0,
        "mid": mid,
        "spread_bps": spread_bps,
    })

def _read_book_l1_resampled_5s(
    book_glob: str,
    so: dict,
    logger: logging.Logger,
    batch_rows: int = 250_000,
    compact_every: int = 200,
) -> pd.DataFrame:
    fs, _, paths = fsspec.get_fs_token_paths(book_glob, storage_options=so)
    paths = sorted(paths)
    if not paths:
        _log(logger, f"[WARN] book glob empty: {book_glob} -> SKIP")
        return pd.DataFrame(columns=["bid0","ask0","mid","spread_bps"])

    out = None
    t0 = time.perf_counter()
    _log(logger, f"book[5s] files={len(paths)} glob={book_glob}")

    appended = 0

    for fi, p in enumerate(paths, 1):
        _log(logger, f"book[5s] open file {fi}/{len(paths)}: {p}")

        with fs.open(p, "rb") as f:
            pf = pq.ParquetFile(f)

            for bi, batch in enumerate(pf.iter_batches(batch_size=batch_rows), 1):
                df = batch.to_pandas()
                if df is None or df.empty:
                    continue

                df = _normalize_book_columns(df)
                df["timestamp"] = _ensure_utc(df["timestamp"])
                df = df.dropna(subset=["timestamp"]).sort_values("timestamp").set_index("timestamp")
                df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["bid0","ask0","mid"])

                g = df[["bid0","ask0","mid","spread_bps"]].resample("5s").last().ffill()
                if not g.empty:
                    for c in ("bid0","ask0","mid","spread_bps"):
                        g[c] = pd.to_numeric(g[c], downcast="float")

                    out = g if out is None else pd.concat([out, g], axis=0)
                    appended += 1

                    if (appended % compact_every) == 0:
                        out = out.sort_index()
                        out = out[~out.index.duplicated(keep="last")]

                del df, g
                if (bi % 10) == 0:
                    mem = _df_mem_mb(out) if out is not None else 0.0
                    _log(logger, f"book[5s] file {fi} batch {bi} appended={appended} mem≈{mem:.1f}MB")
                gc.collect()

        gc.collect()

    if out is None:
        return pd.DataFrame(columns=["bid0","ask0","mid","spread_bps"])

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
    bid_arr: np.ndarray,
    ask_arr: np.ndarray,
    p0: int,
    fill_w: int,
    entry_bid: float,
    entry_ask: float,
) -> Tuple[bool, bool, int]:
    if p0 < 0 or p0 >= bid_arr.size:
        return False, False, -1
    p1 = min(p0 + fill_w, bid_arr.size - 1)
    if p1 <= p0:
        return False, False, -1

    seg_ask = ask_arr[p0:p1+1]
    seg_bid = bid_arr[p0:p1+1]

    iL = _first_true_idx(seg_ask <= entry_bid)   # long maker filled when ask touches down
    iS = _first_true_idx(seg_bid >= entry_ask)   # short maker filled when bid touches up

    okL = (iL >= 0)
    okS = (iS >= 0)
    fill_step = min([x for x in (iL, iS) if x >= 0], default=-1)
    return okL, okS, fill_step

def _rr_label_for_segment_bidask(
    seg_bid: np.ndarray,
    seg_ask: np.ndarray,
    entry_long: float,
    entry_short: float,
    tp_bps: float,
    sl_bps: float,
) -> Tuple[int, str, int]:
    if seg_bid.size < 2 or seg_ask.size < 2:
        return 0, "NONE", -1

    # LONG exits on bid (sell taker)
    tpL_px = entry_long * (1.0 + tp_bps / 1e4)
    slL_px = entry_long * (1.0 - sl_bps / 1e4)
    i_tpL = _first_true_idx(seg_bid >= tpL_px)
    i_slL = _first_true_idx(seg_bid <= slL_px)

    # SHORT exits on ask (buy taker)
    tpS_px = entry_short * (1.0 - tp_bps / 1e4)
    slS_px = entry_short * (1.0 + sl_bps / 1e4)
    i_tpS = _first_true_idx(seg_ask <= tpS_px)
    i_slS = _first_true_idx(seg_ask >= slS_px)

    INF = 10**12
    tpL = i_tpL if i_tpL >= 0 else INF
    slL = i_slL if i_slL >= 0 else INF
    tpS = i_tpS if i_tpS >= 0 else INF
    slS = i_slS if i_slS >= 0 else INF

    if tpL < slL and tpL < INF:
        return 1, "TP_LONG", int(tpL)
    if tpS < slS and tpS < INF:
        return -1, "TP_SHORT", int(tpS)

    return 0, "TIME", -1


# ============================================================
# Core
# ============================================================

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
    # BASE GRID (TF le plus court demandé)
    # -------------------------
    tfs = [str(tf).lower().strip() for tf in (tfs or []) if str(tf).lower().strip() in TF_HORIZONS_SEC]
    if not tfs:
        logger.warning(f"no valid TFs for {symbol} {year}")
        return

    base_tf = _pick_base_tf(tfs)
    base_freq = _to_pandas_freq(base_tf)

    base_book = book_5s[["bid0","ask0","mid","spread_bps"]].resample(base_freq).last().dropna(subset=["bid0","ask0","mid"])
    if base_book.empty:
        logger.warning(f"empty base resample for {symbol} {year} (base_tf={base_tf})")
        return

    base_atr = candles[["atr_bps"]].resample(base_freq).last()

    out = pd.DataFrame(index=base_book.index)
    out.index.name = "t"

    out["t"] = out.index.map(lambda x: pd.Timestamp(x).isoformat().replace("+00:00", "Z"))
    out["symbol"] = symbol
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
    bid_arr = bid_path.to_numpy(dtype="float64", copy=False)
    ask_arr = ask_path.to_numpy(dtype="float64", copy=False)

    pos = bid_path.index.get_indexer(out.index, method="ffill").astype(np.int64, copy=False)
    arr_len = int(bid_arr.size)

    # -------------------------
    # Shared per-row params
    # -------------------------
    step_sec = 5
    fill_w = max(1, int(fill_window_sec // step_sec))

    bid_entry = out["bid_entry"].astype("float64").to_numpy()
    ask_entry = out["ask_entry"].astype("float64").to_numpy()
    cost = out["entry_spread_half_bps"].astype("float64").to_numpy() + float(fee_exit_bps)

    atr_bps_arr = out["atr_bps"].astype("float64").to_numpy()
    R_bps_arr = np.maximum(float(risk_mult) * atr_bps_arr, float(risk_floor_bps)).astype(np.float64)

    n = len(out)

    # -------------------------
    # RR labeling per TF
    # -------------------------
    for tf in tfs:
        horizon_sec = int(TF_HORIZONS_SEC[tf])
        window = int(horizon_sec // step_sec)
        if window < 1:
            window = 1

        if tf in rr_min_map:
            rr_tf = float(rr_min_map[tf])
        elif tf in RR_MIN_MAP:
            rr_tf = float(RR_MIN_MAP[tf])
        else:
            raise ValueError(f"No rr_min defined for TF={tf}")

        TP_bps_arr = (rr_tf * R_bps_arr).astype(np.float64)
        SL_bps_arr = R_bps_arr

        logger.info(f"{symbol} {year} tf={tf} RR start (h={horizon_sec}s window={window} step={step_sec}s rr_min={rr_tf})")
        t_tf0 = _now()

        out[f"horizon_sec_{tf}"] = np.int32(horizon_sec)
        out[f"step_sec_{tf}"] = np.int16(step_sec)
        out[f"rr_min_{tf}"] = np.float32(rr_tf)

        out[f"risk_r_bps_{tf}"] = R_bps_arr.astype("float32")
        out[f"tp_bps_{tf}"] = TP_bps_arr.astype("float32")
        out[f"sl_bps_{tf}"] = SL_bps_arr.astype("float32")

        y = np.zeros(n, dtype=np.int8)
        exit_reason = np.empty(n, dtype=object)
        exit_step = np.full(n, -1, dtype=np.int32)
        pnl_net = np.zeros(n, dtype=np.float32)  # TIME/NOFILL => 0

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

            okL, okS, _ = _maker_fill_ok_conservative(
                bid_arr=bid_arr,
                ask_arr=ask_arr,
                p0=p0,
                fill_w=fill_w,
                entry_bid=eL,
                entry_ask=eS,
            )

            if not okL and not okS:
                y[i] = 0
                exit_reason[i] = "NOFILL"
                exit_step[i] = -1
                pnl_net[i] = 0.0
                continue

            seg_bid = bid_arr[p0:p1+1]
            seg_ask = ask_arr[p0:p1+1]

            tp_bps = float(TP_bps_arr[i])
            sl_bps = float(SL_bps_arr[i])

            yi, rsn, stp = _rr_label_for_segment_bidask(
                seg_bid=seg_bid,
                seg_ask=seg_ask,
                entry_long=eL,
                entry_short=eS,
                tp_bps=tp_bps,
                sl_bps=sl_bps,
            )

            # forbid side if not fillable
            if yi == 1 and (not okL):
                yi, rsn, stp = 0, "NOFILL", -1
            if yi == -1 and (not okS):
                yi, rsn, stp = 0, "NOFILL", -1

            y[i] = yi
            exit_reason[i] = rsn
            exit_step[i] = stp

            if yi == 1 or yi == -1:
                pnl_net[i] = np.float32(tp_bps - cost[i])
            else:
                pnl_net[i] = 0.0

        out[f"y_{tf}"] = y
        out[f"exit_reason_{tf}"] = exit_reason
        out[f"exit_step_{tf}"] = exit_step
        out[f"pnl_net_bps_{tf}"] = pnl_net

        logger.info(f"{symbol} {year} tf={tf} RR done in {_now()-t_tf0:.2f}s")
        gc.collect()

    # -------------------------
    # WRITE CSV
    # -------------------------
    base_cols = [
        "t","symbol","year","symbol_id",
        "bid_entry","ask_entry","mid_entry",
        "spread_bps_entry","atr_bps",
        "fee_exit_bps","entry_spread_half_bps",
        "risk_mult","risk_floor_bps","rr_min_default",
        "fill_mode","fill_window_sec",
    ]

    tf_cols: List[str] = []
    for tf in tfs:
        tf_cols += [
            f"horizon_sec_{tf}",
            f"step_sec_{tf}",
            f"rr_min_{tf}",
            f"risk_r_bps_{tf}",
            f"tp_bps_{tf}",
            f"sl_bps_{tf}",
            f"y_{tf}",
            f"exit_reason_{tf}",
            f"exit_step_{tf}",
            f"pnl_net_bps_{tf}",
        ]

    out_df = out.reset_index(drop=True)[base_cols + tf_cols]

    # free big buffers
    del base_book, base_atr, candles, book_mid_1m, book_5s, bid_path, ask_path, bid_arr, ask_arr, pos
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
        raise ValueError("--rr-min-map-json must be a JSON object like {'15m':2.0,'1h':3.5}")
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

    # ✅ par défaut: sans 5m
    ap.add_argument("--tfs", nargs="+", default=["15m","30m","1h","2h","4h"])

    ap.add_argument("--aws-region", default="ap-northeast-1")

    ap.add_argument("--risk-mult", type=float, default=1.0, help="R = risk_mult * ATR_bps (before floor)")
    ap.add_argument("--rr-min-map-json", type=str, default=None,
                    help="Override rr_min per TF. Example: '{\"15m\":2.0,\"30m\":3.0,\"1h\":3.5}'")
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