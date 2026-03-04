#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ml_bot/core/stage0/spr_stream_250ms.py

from __future__ import annotations

import gc
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")

def log(msg: str) -> None:
    print(f"{_now()} {msg}", flush=True)

def to_utc_ts(x) -> pd.Series:
    """Coerce to tz-aware UTC timestamps."""
    return pd.to_datetime(x, errors="coerce", utc=True)

def bucket_right_edge_ms(ts: pd.Series, freq_ms: int) -> np.ndarray:
    """
    Right-edge bucketing in ns (int64).
    bucket_end = ((t // step) + 1) * step
    """
    ts = to_utc_ts(ts)
    ns = ts.astype("int64").to_numpy()
    step = np.int64(freq_ms) * np.int64(1_000_000)  # ms -> ns
    return ((ns // step) + 1) * step

def _to_arr_f32(x, n: int) -> np.ndarray:
    """
    Accepts pandas Series/array OR scalar.
    Returns float32 numpy array of length n.
    """
    if isinstance(x, (pd.Series, np.ndarray, list, tuple)):
        return pd.to_numeric(x, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    # scalar fallback
    return np.full(n, float(x), dtype=np.float32)

def _to_arr_f64(x, n: int) -> np.ndarray:
    """
    Accepts pandas Series/array OR scalar.
    Returns float64 numpy array of length n.
    """
    if isinstance(x, (pd.Series, np.ndarray, list, tuple)):
        return pd.to_numeric(x, errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    return np.full(n, float(x), dtype=np.float64)

def stream_book_last_month_L15_ordered(
    path: str,
    freq_ms: int = 250,
    max_level: int = 15,
    batch_rows: int = 300_000,
    log_every_batches: int = 20,
) -> pd.DataFrame:
    """
    Stream book parquet and keep last snapshot per bucket (RIGHT EDGE) using
    a single-pass ordered approach (no dict per bucket).

    Assumption: parquet rows are roughly ordered by timestamp.
    If not perfectly ordered, this still works "well enough" for your stage0,
    but worst-case could miss the true last snapshot inside the bucket if data is shuffled.

    Output: DataFrame with columns:
      timestamp (bucket_end), bid_0_price, ask_0_price, bid_0_size, ask_0_size,
      bid_i_size / ask_i_size for i=0..max_level
    """
    t0 = time.time()
    log(f"[book] open {path}")

    pf = pq.ParquetFile(path)
    schema = set(pf.schema.names)
    log(f"[book] max_level={max_level} present_levels=" +
        ",".join(str(i) for i in range(max_level+1) if f"bid_{i}_size" in schema))

    # required
    base_cols = ["timestamp", "bid_0_price", "ask_0_price", "bid_0_size", "ask_0_size"]
    size_cols = []
    for i in range(max_level + 1):
        size_cols += [f"bid_{i}_size", f"ask_{i}_size"]

    cols = [c for c in (base_cols + size_cols) if c in schema]
    if "timestamp" not in cols or "bid_0_price" not in cols or "ask_0_price" not in cols:
        raise ValueError(f"Book missing required columns in {path}")

    # Output buffers
    out_bucket_ns: List[int] = []
    out_bid0: List[float] = []
    out_ask0: List[float] = []
    out_bsz0: List[float] = []
    out_asz0: List[float] = []
    out_bid_sizes = [ [] for _ in range(max_level + 1) ]
    out_ask_sizes = [ [] for _ in range(max_level + 1) ]

    # Current bucket state (last snapshot inside current bucket)
    cur_bucket: Optional[int] = None
    cur_bid0 = np.nan
    cur_ask0 = np.nan
    cur_bsz0 = 0.0
    cur_asz0 = 0.0
    cur_bid_arr = np.zeros(max_level + 1, dtype=np.float32)
    cur_ask_arr = np.zeros(max_level + 1, dtype=np.float32)

    def flush_cur():
        # skip invalid
        if cur_bucket is None:
            return
        if not np.isfinite(cur_bid0) or not np.isfinite(cur_ask0) or cur_bid0 <= 0 or cur_ask0 <= 0:
            return
        out_bucket_ns.append(int(cur_bucket))
        out_bid0.append(float(cur_bid0))
        out_ask0.append(float(cur_ask0))
        out_bsz0.append(float(cur_bsz0))
        out_asz0.append(float(cur_asz0))
        for i in range(max_level + 1):
            out_bid_sizes[i].append(float(cur_bid_arr[i]))
            out_ask_sizes[i].append(float(cur_ask_arr[i]))

    rows = 0
    kept = 0
    batches = 0

    for batch in pf.iter_batches(batch_size=batch_rows, columns=cols):
        batches += 1
        df = batch.to_pandas()

        ts = to_utc_ts(df["timestamp"])
        ok = ~ts.isna()
        if not ok.any():
            del df
            gc.collect()
            continue

        df = df.loc[ok]
        ts = ts.loc[df.index]

        bucket_ns = bucket_right_edge_ms(ts, freq_ms=freq_ms)
        rows += len(df)

        bid0 = pd.to_numeric(df["bid_0_price"], errors="coerce").to_numpy(dtype=np.float64)
        ask0 = pd.to_numeric(df["ask_0_price"], errors="coerce").to_numpy(dtype=np.float64)
        bsz0 = pd.to_numeric(df.get("bid_0_size", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        asz0 = pd.to_numeric(df.get("ask_0_size", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)

        # size arrays
        nrows = len(df)

        bid_sizes = []
        ask_sizes = []
        for i in range(max_level + 1):
            bid_sizes.append(_to_arr_f32(df.get(f"bid_{i}_size", 0.0), nrows))
            ask_sizes.append(_to_arr_f32(df.get(f"ask_{i}_size", 0.0), nrows))

        # single pass through rows
        for j in range(len(df)):
            bkt = int(bucket_ns[j])

            # bucket changed => flush previous bucket
            if cur_bucket is None:
                cur_bucket = bkt
            elif bkt != cur_bucket:
                flush_cur()
                kept += 1
                cur_bucket = bkt

            # always overwrite current snapshot (we want last one in bucket)
            cur_bid0 = bid0[j]
            cur_ask0 = ask0[j]
            cur_bsz0 = bsz0[j]
            cur_asz0 = asz0[j]
            for i in range(max_level + 1):
                cur_bid_arr[i] = bid_sizes[i][j]
                cur_ask_arr[i] = ask_sizes[i][j]

        del df
        gc.collect()

        if (batches % log_every_batches) == 0:
            elapsed = time.time() - t0
            log(f"[book] batches={batches} rows={rows:,} buckets_out={len(out_bucket_ns):,} elapsed={elapsed:.1f}s")

    # flush last bucket
    flush_cur()
    kept += 1

    elapsed = time.time() - t0
    log(f"[book] done: buckets={len(out_bucket_ns):,} elapsed={elapsed:.1f}s")

    if len(out_bucket_ns) == 0:
        return pd.DataFrame()

    idx = pd.to_datetime(np.array(out_bucket_ns, dtype=np.int64), utc=True)
    out = pd.DataFrame({"timestamp": idx})
    out["bid_0_price"] = np.asarray(out_bid0, dtype=np.float64)
    out["ask_0_price"] = np.asarray(out_ask0, dtype=np.float64)
    out["bid_0_size"] = np.asarray(out_bsz0, dtype=np.float32)
    out["ask_0_size"] = np.asarray(out_asz0, dtype=np.float32)

    for i in range(max_level + 1):
        out[f"bid_{i}_size"] = np.asarray(out_bid_sizes[i], dtype=np.float32)
        out[f"ask_{i}_size"] = np.asarray(out_ask_sizes[i], dtype=np.float32)

    return out

def stream_trades_bucket_month_ordered(
    path: str,
    freq_ms: int = 250,
    batch_rows: int = 300_000,
    log_every_batches: int = 20,
) -> pd.DataFrame:
    """
    Stream trades parquet and aggregate per bucket (RIGHT EDGE) with ordered single-pass.

    Output indexed by bucket_end timestamp with columns:
      - notional_buy, notional_sell, ntr
      - (optional) qty_buy, qty_sell
    """
    t0 = time.time()
    log(f"[trades] open {path}")

    pf = pq.ParquetFile(path)
    schema = set(pf.schema.names)

    if "timestamp" not in schema:
        raise ValueError(f"Trades missing timestamp in {path}")

    qty_col = "qty" if "qty" in schema else ("quantity" if "quantity" in schema else None)
    if qty_col is None:
        raise ValueError(f"Trades missing qty/quantity in {path}")

    if "price" not in schema:
        raise ValueError(f"Trades missing price in {path}")

    has_is_aggr_buy = "is_aggr_buy" in schema
    has_side = "side" in schema

    cols = ["timestamp", qty_col, "price"]
    if has_is_aggr_buy:
        cols.append("is_aggr_buy")
    elif has_side:
        cols.append("side")
    else:
        raise ValueError(f"Trades missing direction (is_aggr_buy or side) in {path}")

    out_bucket_ns: List[int] = []
    out_nb: List[float] = []
    out_ns: List[float] = []
    out_ntr: List[int] = []

    # optional debug / future use
    out_qb: List[float] = []
    out_qs: List[float] = []

    cur_bucket: Optional[int] = None
    cur_nb = 0.0
    cur_ns = 0.0
    cur_qb = 0.0
    cur_qs = 0.0
    cur_ntr = 0

    def flush_cur():
        if cur_bucket is None:
            return
        out_bucket_ns.append(int(cur_bucket))
        out_nb.append(float(cur_nb))
        out_ns.append(float(cur_ns))
        out_qb.append(float(cur_qb))
        out_qs.append(float(cur_qs))
        out_ntr.append(int(cur_ntr))

    rows = 0
    batches = 0

    for batch in pf.iter_batches(batch_size=batch_rows, columns=cols):
        batches += 1
        df = batch.to_pandas()

        ts = to_utc_ts(df["timestamp"])
        ok = ~ts.isna()
        if not ok.any():
            del df
            gc.collect()
            continue

        df = df.loc[ok]
        ts = ts.loc[df.index]

        bucket_ns = bucket_right_edge_ms(ts, freq_ms=freq_ms)
        rows += len(df)

        qty = pd.to_numeric(df[qty_col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        px  = pd.to_numeric(df["price"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        notional = qty * px

        if has_is_aggr_buy:
            is_buy = df["is_aggr_buy"].astype(bool).to_numpy()
        else:
            side = df["side"].astype(str).str.lower().to_numpy()
            is_buy = (side == "buy")

        for j in range(len(df)):
            bkt = int(bucket_ns[j])

            if cur_bucket is None:
                cur_bucket = bkt
            elif bkt != cur_bucket:
                flush_cur()
                cur_bucket = bkt
                cur_nb = 0.0
                cur_ns = 0.0
                cur_qb = 0.0
                cur_qs = 0.0
                cur_ntr = 0

            q = float(qty[j])
            n = float(notional[j])

            if bool(is_buy[j]):
                cur_qb += q
                cur_nb += n
            else:
                cur_qs += q
                cur_ns += n
            cur_ntr += 1

        del df
        gc.collect()

        if (batches % log_every_batches) == 0:
            elapsed = time.time() - t0
            log(f"[trades] batches={batches} rows={rows:,} buckets_out={len(out_bucket_ns):,} elapsed={elapsed:.1f}s")

    flush_cur()
    elapsed = time.time() - t0
    log(f"[trades] done: buckets={len(out_bucket_ns):,} elapsed={elapsed:.1f}s")

    if len(out_bucket_ns) == 0:
        return pd.DataFrame()

    idx = pd.to_datetime(np.array(out_bucket_ns, dtype=np.int64), utc=True)
    out = pd.DataFrame(index=idx)
    out["notional_buy"] = np.asarray(out_nb, dtype=np.float64)
    out["notional_sell"] = np.asarray(out_ns, dtype=np.float64)
    out["qty_buy"] = np.asarray(out_qb, dtype=np.float64)
    out["qty_sell"] = np.asarray(out_qs, dtype=np.float64)
    out["ntr"] = np.asarray(out_ntr, dtype=np.int32)
    out = out.sort_index()
    return out

def build_stage0_features_streaming_250ms(
    book_path: str,
    trades_path: str,
    cfg,
    freq_ms: int = 250,
    batch_rows: int = 300_000,
    max_level: int = 15,
) -> pd.DataFrame:
    """
    Build stage0 features on 250ms buckets using L15 book + trades bucketed,
    then compute rolling trade-flow, 10m range, etc.

    NOTE: This function intentionally does NOT compute OBI/Dbid/Dask/micro.
    Those are computed in spr_v1.compute_book_core_features after we join.
    """
    t0 = time.time()
    log("[features] start")

    # 1) stream book last snapshot per bucket (L15)
    book_b = stream_book_last_month_L15_ordered(
        book_path, freq_ms=freq_ms, max_level=max_level, batch_rows=batch_rows
    )
    if book_b is None or len(book_b) == 0:
        return pd.DataFrame()

    # 2) stream trades aggregated per bucket
    tr_b = stream_trades_bucket_month_ordered(
        trades_path, freq_ms=freq_ms, batch_rows=batch_rows
    )

    # 3) join
    book_b = book_b.copy()
    book_b["timestamp"] = to_utc_ts(book_b["timestamp"])
    book_b = book_b.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    book_b = book_b.set_index("timestamp")

    if tr_b is None or len(tr_b) == 0:
        tr_b = pd.DataFrame(index=pd.to_datetime([], utc=True),
                            columns=["notional_buy", "notional_sell", "qty_buy", "qty_sell", "ntr"],
                            )

    tr_b = tr_b.sort_index()

    log(f"[features] join book={len(book_b):,} trades_buckets={len(tr_b):,}")
    feats = book_b.join(tr_b, how="left")

    # après join
    for c in ["notional_buy", "notional_sell"]:
        if c not in feats.columns:
            feats[c] = 0.0
        feats[c] = feats[c].fillna(0.0).astype(np.float64)

    for c in ["qty_buy", "qty_sell"]:
        if c not in feats.columns:
            feats[c] = 0.0
        feats[c] = feats[c].fillna(0.0).astype(np.float32)

    feats["ntr"] = feats["ntr"].fillna(0).astype(np.int32)

    # --- rolling notional flow (2s window by default)
    feats = feats.sort_index()  # index = timestamp bucket_end

    step_s = freq_ms / 1000.0
    win_buckets = max(1, int(round(cfg.trades_window_s / step_s)))  # 2s / 0.25s = 8

    Nb = feats["notional_buy"].rolling(win_buckets, min_periods=1).sum()
    Ns = feats["notional_sell"].rolling(win_buckets, min_periods=1).sum()
    ntr_roll = feats["ntr"].rolling(win_buckets, min_periods=1).sum()

    feats["Nb"] = Nb
    feats["Ns"] = Ns
    feats["Ntot"] = feats["Nb"] + feats["Ns"]

    # TI sur notional (on garde)
    feats["TI"] = (feats["Nb"] - feats["Ns"]) / (feats["Ntot"] + 1e-9)

    # ✅ vrai nps = trades/sec
    feats["nps"] = ntr_roll / float(cfg.trades_window_s)

    # ✅ debug only: notional/sec (USDT/s) — not used in filters
    feats["notional_ps_dbg"] = feats["Ntot"] / float(cfg.trades_window_s)

    feats = feats.reset_index()

    elapsed = time.time() - t0
    log(f"[features] done join rows={len(feats):,} elapsed={elapsed:.1f}s")
    return feats