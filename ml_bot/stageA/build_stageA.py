#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
stageA.build_stageA.py

Build StageA dataset on S3:
- symbol: BTCUSDT (but CLI supports list)
- decision grid: every 30s
- horizon: 2 minutes (120s)
- confirmation: +20s rule-based (pure rule, configurable)
- label: Label A (TP/SL barrier, conservative maker fill proxy, max duration 120s)
- features V1: candles 1m + book L15 + trades

Inputs:
--book-root   s3://.../data/book/<SYMBOL>/YYYY-MM.parquet
--trade-root  s3://.../data/trade/<SYMBOL>/YYYY-MM.parquet
--candle-root s3://.../data/bougie/<SYMBOL>/<SYMBOL>-1m-YYYY.parquet
Output:
--out-root    s3://.../data/stageA/symbol=.../year=YYYY/month=MM/stageA_btc_2m_30s.parquet
Also writes:
--out-root/schemaA.json
"""

from __future__ import annotations
from datetime import datetime, timezone

import argparse
import json
import os
import sys
import gc
import logging
import time
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import fsspec
import pyarrow as pa
import pyarrow.parquet as pq


# ---------------------------------------------------------------------
# You said you already pasted build_stageA_schema() in this file.
# Keep it above, and it must return the schema dict.
# ---------------------------------------------------------------------
def build_stageA_schema():
    # Schéma "contract" (StageA BTC 2m horizon, décision 30s, confirm 20s)
    schema = {
        "stage": "stageA",
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "grain": "1 row per decision timestamp",
        "defaults": {
            "cfg_horizon_sec": 120,
            "cfg_decision_step_sec": 30,
            "cfg_k_confirm_sec": 20,
            "id_symbol": "BTCUSDT",
        },
        "column_groups": {
            "id": [
                {"name": "id_row_id", "dtype": "string"},
                {"name": "id_symbol", "dtype": "string"},
                {"name": "id_t", "dtype": "datetime64[ns, UTC]"},
                {"name": "id_year", "dtype": "int16"},
                {"name": "id_month", "dtype": "int8"},
            ],
            "label": [
                {"name": "label_A", "dtype": "int8", "allowed": [0, 1]},
                {"name": "label_A_reason", "dtype": "string"},
                {"name": "label_A_exit_reason", "dtype": "string"},
                {"name": "label_A_pnl_net_bps", "dtype": "float32"},
                {"name": "label_A_exit_t_sec", "dtype": "int16"},
            ],
            "cfg": [
                {"name": "cfg_horizon_sec", "dtype": "int16"},
                {"name": "cfg_decision_step_sec", "dtype": "int16"},
                {"name": "cfg_k_confirm_sec", "dtype": "int16"},
                {"name": "cfg_fee_exit_bps", "dtype": "float32"},
                {"name": "cfg_cost_model", "dtype": "string"},
            ],
            "audit": [
                {"name": "audit_spread_bps_entry", "dtype": "float32"},
                {"name": "audit_cost_bps", "dtype": "float32"},
                {"name": "audit_R_bps", "dtype": "float32"},
                {"name": "audit_tp_bps", "dtype": "float32"},
                {"name": "audit_sl_bps", "dtype": "float32"},
                {"name": "audit_rr_min", "dtype": "float32"},
                {"name": "audit_cost_R", "dtype": "float32"},
                {"name": "audit_p_thr_ev0", "dtype": "float32"},
            ],
            "features": [
                # ---- candles 1m ----
                {"name": "f_c_ret_1m_bps", "dtype": "float32"},
                {"name": "f_c_ret_2m_bps", "dtype": "float32"},
                {"name": "f_c_absret_2m_bps", "dtype": "float32"},
                {"name": "f_c_absret_5m_bps", "dtype": "float32"},
                {"name": "f_c_absret_15m_bps", "dtype": "float32"},
                {"name": "f_c_range_1m_bps", "dtype": "float32"},
                {"name": "f_c_range_2m_bps", "dtype": "float32"},
                {"name": "f_c_body_to_range_1m", "dtype": "float32"},
                {"name": "f_c_wick_ratio_1m", "dtype": "float32"},
                {"name": "f_c_vol_5m_bps", "dtype": "float32"},
                {"name": "f_c_vol_15m_bps", "dtype": "float32"},
                {"name": "f_c_ema_gap_3_9_bps", "dtype": "float32"},
                {"name": "f_c_slope_ema9_bps", "dtype": "float32"},

                # ---- book L15 ----
                {"name": "f_b_spread_bps", "dtype": "float32"},
                {"name": "f_b_mid_move_10s_bps", "dtype": "float32"},
                {"name": "f_b_mid_move_30s_bps", "dtype": "float32"},
                {"name": "f_b_imb_L1", "dtype": "float32"},
                {"name": "f_b_imb_L5", "dtype": "float32"},
                {"name": "f_b_imb_L15", "dtype": "float32"},
                {"name": "f_b_depth_bid_L15", "dtype": "float32"},
                {"name": "f_b_depth_ask_L15", "dtype": "float32"},
                {"name": "f_b_depth_ratio_L15", "dtype": "float32"},
                {"name": "f_b_slope_bid_L15", "dtype": "float32"},
                {"name": "f_b_slope_ask_L15", "dtype": "float32"},
                {"name": "f_b_quote_updates_10s", "dtype": "int32"},
                {"name": "f_b_quote_updates_30s", "dtype": "int32"},
                {"name": "f_b_spread_vol_30s", "dtype": "float32"},

                # ---- trades ----
                {"name": "f_t_aggr_buy_ratio_10s", "dtype": "float32"},
                {"name": "f_t_aggr_buy_ratio_30s", "dtype": "float32"},
                {"name": "f_t_aggr_buy_ratio_120s", "dtype": "float32"},
                {"name": "f_t_signed_vol_10s", "dtype": "float32"},
                {"name": "f_t_signed_vol_30s", "dtype": "float32"},
                {"name": "f_t_signed_vol_120s", "dtype": "float32"},
                {"name": "f_t_trade_count_10s", "dtype": "int32"},
                {"name": "f_t_trade_count_30s", "dtype": "int32"},
                {"name": "f_t_trade_count_120s", "dtype": "int32"},
                {"name": "f_t_vol_10s", "dtype": "float32"},
                {"name": "f_t_vol_30s", "dtype": "float32"},
                {"name": "f_t_vol_120s", "dtype": "float32"},
                {"name": "f_t_burst_trade_count", "dtype": "float32"},
                {"name": "f_t_burst_vol", "dtype": "float32"},

                # ---- cross (book x trades) ----
                {"name": "f_x_flow_align_10s", "dtype": "float32"},
                {"name": "f_x_liquidity_stress", "dtype": "float32"},
                {"name": "f_x_impact_proxy_10s", "dtype": "float32"},
            ],
        },
        "rules": {
            "features_prefix": "f_",
            "no_leakage": [
                "No feature may use data > id_t.",
                "Columns starting with label_ or audit_ or cfg_ must NOT be used as ML inputs.",
            ],
            "ranges_sanity": {
                "audit_p_thr_ev0": [0.0, 1.0],
                "label_A_exit_t_sec": [-1, 120],
            }
        }
    }

    # Flatten list of all columns (order = id -> label -> cfg -> audit -> features)
    ordered = []
    for grp in ["id", "label", "cfg", "audit", "features"]:
        ordered.extend(schema["column_groups"][grp])
    schema["all_columns_ordered"] = ordered
    return schema


# =========================
# Logging
# =========================
def setup_logger(verbose: bool = True) -> logging.Logger:
    logger = logging.getLogger("stageA")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO if verbose else logging.WARNING)
    h = logging.StreamHandler(stream=sys.stdout)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    h.setFormatter(fmt)
    logger.addHandler(h)
    return logger

def log(logger: logging.Logger, msg: str):
    logger.info(msg)
    for h in logger.handlers:
        try:
            h.flush()
        except Exception:
            pass
    sys.stdout.flush()
    sys.stderr.flush()

def ensure_utc(ts):
    return pd.to_datetime(ts, utc=True, errors="coerce")

def s3_so(region: Optional[str]) -> dict:
    return {"client_kwargs": {"region_name": region}} if region else {}

def exists(path: str, so: dict) -> bool:
    fs, _, _ = fsspec.get_fs_token_paths(path, storage_options=so)
    return fs.exists(path)

def row_id(symbol: str, t_iso: str, salt: str = "stageA") -> str:
    key = f"{symbol}|{t_iso}|{salt}".encode("utf-8")
    return hashlib.blake2b(key, digest_size=16).hexdigest()

def month_iter(years: List[int], months: Optional[List[int]]) -> List[Tuple[int,int]]:
    out = []
    for y in years:
        if months:
            for m in months:
                out.append((y, m))
        else:
            for m in range(1, 13):
                out.append((y, m))
    return out

def yyyy_mm(y: int, m: int) -> str:
    return f"{y:04d}-{m:02d}"

def out_partition(out_root: str, symbol: str, y: int, m: int) -> str:
    return f"{out_root.rstrip('/')}/symbol={symbol}/year={y:04d}/month={m:02d}"

def safe_mkdir(path: str, so: dict):
    if path.startswith("s3://"):
        fs = fsspec.filesystem("s3", **so)
        if not fs.exists(path):
            fs.mkdirs(path, exist_ok=True)
    else:
        Path(path).mkdir(parents=True, exist_ok=True)

def write_json(path: str, obj: dict, so: dict):
    payload = json.dumps(obj, indent=2, ensure_ascii=False).encode("utf-8")
    if path.startswith("s3://"):
        with fsspec.open(path, "wb", **so) as f:
            f.write(payload)
    else:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(payload)

def write_parquet(path: str, df: pd.DataFrame, so: dict, compression: str = "snappy"):
    table = pa.Table.from_pandas(df, preserve_index=False)
    if path.startswith("s3://"):
        with fsspec.open(path, "wb", **so) as f:
            pq.write_table(table, f, compression=compression)
    else:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, path, compression=compression)

def write_parquet_part(
    out_dir: str,
    part_id: int,
    df: pd.DataFrame,
    so: dict,
    compression: str = "snappy",
):
    fname = f"part-{part_id:05d}.parquet"
    path = f"{out_dir.rstrip('/')}/{fname}"
    write_parquet(path, df, so, compression=compression)
    return path

# =========================
# ATR on candles 1m
# =========================
def atr_ewm(c: pd.DataFrame, n: int = 14) -> pd.Series:
    h = c["high"].astype(float)
    l = c["low"].astype(float)
    close = c["close"].astype(float)
    pc = close.shift(1)
    tr = pd.concat([(h-l).abs(), (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

# =========================
# Trades normalization
# =========================
def normalize_trades(df: pd.DataFrame) -> pd.DataFrame:
    # Expected raw cols example:
    # ['trade_id','price','quantity','origin_time','side','received_time','symbol','exchange','dt']
    out = pd.DataFrame()
    if "origin_time" in df.columns:
        out["timestamp"] = ensure_utc(df["origin_time"])
    elif "timestamp" in df.columns:
        out["timestamp"] = ensure_utc(df["timestamp"])
    else:
        raise ValueError("Trades missing origin_time/timestamp")

    out["price"] = pd.to_numeric(df.get("price", np.nan), errors="coerce")
    qty_col = "quantity" if "quantity" in df.columns else ("qty" if "qty" in df.columns else None)
    if qty_col is None:
        raise ValueError("Trades missing quantity/qty")
    out["qty"] = pd.to_numeric(df[qty_col], errors="coerce").fillna(0.0)

    side = df.get("side", None)
    if side is None:
        # fallback: assume unknown
        out["is_aggr_buy"] = False
    else:
        s = side.astype(str).str.lower().str.strip()
        # In your sample: side is 'buy'/'sell'
        out["is_aggr_buy"] = (s == "buy")

    out = out.dropna(subset=["timestamp"]).sort_values("timestamp")
    out = out[(out["qty"] > 0) & np.isfinite(out["price"])]
    out = out.set_index("timestamp", drop=True)
    out = out[~out.index.duplicated(keep="last")]
    return out

# =========================
# Book reading + normalize (L15)
# =========================
def normalize_book_l15(df: pd.DataFrame) -> pd.DataFrame:
    # Accept schema like bid_0_price/ask_0_price + sizes
    ts_col = "timestamp" if "timestamp" in df.columns else None
    if ts_col is None:
        raise ValueError("Book missing timestamp")

    out = df.copy()
    out[ts_col] = ensure_utc(out[ts_col])
    out = out.dropna(subset=[ts_col]).sort_values(ts_col)
    out = out.set_index(ts_col, drop=True)

    # Identify top of book
    bid0 = None
    ask0 = None
    for b, a in [("bid_0_price", "ask_0_price"), ("bid0", "ask0"), ("bid", "ask")]:
        if b in out.columns and a in out.columns:
            bid0, ask0 = b, a
            break
    if bid0 is None:
        raise ValueError("Book missing bid0/ask0 price columns")

    out["bid0"] = pd.to_numeric(out[bid0], errors="coerce")
    out["ask0"] = pd.to_numeric(out[ask0], errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["bid0", "ask0"])
    out = out[(out["bid0"] > 0) & (out["ask0"] > 0) & (out["ask0"] > out["bid0"])]

    out["mid"] = (out["bid0"].astype("float64") + out["ask0"].astype("float64")) / 2.0
    out["spread_bps"] = ((out["ask0"].astype("float64") - out["bid0"].astype("float64")) / out["mid"].astype("float64")) * 1e4

    # sizes levels (optional)
    # We accept bid_i_size / ask_i_size, i=0..14
    # If not present, some features will become NaN.
    return out

# =========================
# Feature engineering
# =========================
def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def candle_features_1m(candles_1m: pd.DataFrame, t_index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Compute candle features and align to decision timestamps (t_index).
    Uses only info <= t (no leakage).
    """
    c = candles_1m.copy()
    c = c.sort_index()
    close = c["close"].astype("float64")
    open_ = c["open"].astype("float64")
    high = c["high"].astype("float64")
    low  = c["low"].astype("float64")

    ret1m_bps = (close.pct_change() * 1e4)
    ret2m_bps = (close.pct_change(2) * 1e4)

    rng1m_bps = ((high - low) / close.replace(0, np.nan) * 1e4)
    body = (close - open_).abs()
    rng = (high - low).replace(0, np.nan)
    body_to_range = (body / rng)
    wick_ratio = ((rng - body) / rng).clip(lower=0)

    # absret windows
    absret_2m_bps = ret2m_bps.abs()
    absret_5m_bps = (close.pct_change(5) * 1e4).abs()
    absret_15m_bps = (close.pct_change(15) * 1e4).abs()

    # vol proxy (rolling std of 1m returns)
    vol_5m_bps = ret1m_bps.rolling(5, min_periods=3).std()
    vol_15m_bps = ret1m_bps.rolling(15, min_periods=5).std()

    ema3 = _ema(close, 3)
    ema9 = _ema(close, 9)
    ema_gap_3_9_bps = ((ema3 - ema9) / close.replace(0, np.nan) * 1e4)
    slope_ema9_bps = (ema9.diff() / close.replace(0, np.nan) * 1e4)

    feat = pd.DataFrame({
        "f_c_ret_1m_bps": ret1m_bps,
        "f_c_ret_2m_bps": ret2m_bps,
        "f_c_absret_2m_bps": absret_2m_bps,
        "f_c_absret_5m_bps": absret_5m_bps,
        "f_c_absret_15m_bps": absret_15m_bps,
        "f_c_range_1m_bps": rng1m_bps,
        "f_c_range_2m_bps": ((high.rolling(2).max() - low.rolling(2).min()) / close.replace(0, np.nan) * 1e4),
        "f_c_body_to_range_1m": body_to_range,
        "f_c_wick_ratio_1m": wick_ratio,
        "f_c_vol_5m_bps": vol_5m_bps,
        "f_c_vol_15m_bps": vol_15m_bps,
        "f_c_ema_gap_3_9_bps": ema_gap_3_9_bps,
        "f_c_slope_ema9_bps": slope_ema9_bps,
    }, index=c.index)

    # align to decision timestamps using last available 1m candle <= t
    # decision timestamps every 30s: use asof join (ffill on time)
    feat = feat.reindex(t_index, method="ffill")
    return feat.astype("float32")

def book_depth_features(book_1s: pd.DataFrame, t_index: pd.DatetimeIndex, L: int = 15) -> pd.DataFrame:
    b1 = book_1s
    if b1 is None or b1.empty:
        cols = [
            "f_b_spread_bps","f_b_mid_move_10s_bps","f_b_mid_move_30s_bps",
            "f_b_imb_L1","f_b_imb_L5","f_b_imb_L15",
            "f_b_depth_bid_L15","f_b_depth_ask_L15","f_b_depth_ratio_L15",
            "f_b_slope_bid_L15","f_b_slope_ask_L15",
            "f_b_quote_updates_10s","f_b_quote_updates_30s","f_b_spread_vol_30s",
        ]
        out = pd.DataFrame(index=t_index, data={c: np.nan for c in cols})
        out["f_b_quote_updates_10s"] = 0
        out["f_b_quote_updates_30s"] = 0
        return out

    if not b1.index.is_monotonic_increasing:
        b1 = b1.sort_index()

    mid = b1["mid"].astype("float64")
    spread = b1["spread_bps"].astype("float64")

    mid_move_10s = (mid.pct_change(10) * 1e4)
    mid_move_30s = (mid.pct_change(30) * 1e4)

    ch = (b1["bid0"].diff().ne(0) | b1["ask0"].diff().ne(0)).astype("int32")
    q_updates_10s = ch.rolling(10, min_periods=1).sum()
    q_updates_30s = ch.rolling(30, min_periods=1).sum()

    spread_vol_30s = spread.rolling(30, min_periods=5).std()

    # --- helper pour récupérer size col ---
    def _colname(i: int, side: str) -> Optional[str]:
        # prioriser _size, fallback _qty
        c1 = f"{side}_{i}_size"
        c2 = f"{side}_{i}_qty"
        if c1 in b1.columns: return c1
        if c2 in b1.columns: return c2
        return None

    bid_cols = [_colname(i,"bid") for i in range(L)]
    ask_cols = [_colname(i,"ask") for i in range(L)]

    # juste après bid_cols / ask_cols
    bid_cols_all = [c for c in bid_cols if c is not None]
    ask_cols_all = [c for c in ask_cols if c is not None]

    bid_mat = None
    ask_mat = None
    if bid_cols_all:
        bid_mat = b1[bid_cols_all].fillna(0.0).to_numpy(dtype=np.float32, copy=False)
    if ask_cols_all:
        ask_mat = b1[ask_cols_all].fillna(0.0).to_numpy(dtype=np.float32, copy=False)

    def _sum_depth_mat(mat, k: int) -> pd.Series:
        if mat is None or mat.shape[1] < 1:
            return pd.Series(np.nan, index=b1.index)
        kk = min(k, mat.shape[1])
        return pd.Series(mat[:, :kk].sum(axis=1), index=b1.index, dtype="float32")

    bid_L1  = _sum_depth_mat(bid_mat, 1)
    ask_L1  = _sum_depth_mat(ask_mat, 1)
    bid_L5  = _sum_depth_mat(bid_mat, 5)
    ask_L5  = _sum_depth_mat(ask_mat, 5)
    bid_L15 = _sum_depth_mat(bid_mat, 15)
    ask_L15 = _sum_depth_mat(ask_mat, 15)

    def _imb(bid_sum: pd.Series, ask_sum: pd.Series) -> pd.Series:
        den = (bid_sum + ask_sum).replace(0, np.nan)
        return ((bid_sum - ask_sum) / den)

    imb_L1  = _imb(bid_L1,  ask_L1)
    imb_L5  = _imb(bid_L5,  ask_L5)
    imb_L15 = _imb(bid_L15, ask_L15)

    depth_bid_L15 = bid_L15
    depth_ask_L15 = ask_L15
    depth_ratio_L15 = (bid_L15 / ask_L15.replace(0, np.nan))

    # slopes simples (premium “ok” sans être parfait):
    # pente = (sum first 5 - sum last 5) / sum total
    bid_last5 = bid_L15 - _sum_depth_mat(bid_mat, 10)
    ask_last5 = ask_L15 - _sum_depth_mat(ask_mat, 10)

    bid_first5 = bid_L5
    ask_first5 = ask_L5
    
    slope_bid = (bid_first5 - bid_last5) / bid_L15.replace(0, np.nan)
    slope_ask = (ask_first5 - ask_last5) / ask_L15.replace(0, np.nan)

    feat = pd.DataFrame({
        "f_b_spread_bps": spread,
        "f_b_mid_move_10s_bps": mid_move_10s,
        "f_b_mid_move_30s_bps": mid_move_30s,
        "f_b_imb_L1": imb_L1,
        "f_b_imb_L5": imb_L5,
        "f_b_imb_L15": imb_L15,
        "f_b_depth_bid_L15": depth_bid_L15,
        "f_b_depth_ask_L15": depth_ask_L15,
        "f_b_depth_ratio_L15": depth_ratio_L15,
        "f_b_slope_bid_L15": slope_bid,
        "f_b_slope_ask_L15": slope_ask,
        "f_b_quote_updates_10s": q_updates_10s,
        "f_b_quote_updates_30s": q_updates_30s,
        "f_b_spread_vol_30s": spread_vol_30s,
    }, index=b1.index)

    feat = feat.reindex(t_index, method="ffill")
    feat = feat.replace([np.inf, -np.inf], np.nan)

    feat["f_b_quote_updates_10s"] = feat["f_b_quote_updates_10s"].fillna(0).astype("int32")
    feat["f_b_quote_updates_30s"] = feat["f_b_quote_updates_30s"].fillna(0).astype("int32")

    # float32 partout
    for c in feat.columns:
        if c.startswith("f_b_") and c not in ["f_b_quote_updates_10s","f_b_quote_updates_30s"]:
            feat[c] = feat[c].astype("float32")

    return feat

def trades_features(tr: pd.DataFrame, t_index: pd.DatetimeIndex) -> pd.DataFrame:
    if tr is None or tr.empty:
        cols_f = [
            "f_t_aggr_buy_ratio_10s","f_t_aggr_buy_ratio_30s","f_t_aggr_buy_ratio_120s",
            "f_t_signed_vol_10s","f_t_signed_vol_30s","f_t_signed_vol_120s",
            "f_t_trade_count_10s","f_t_trade_count_30s","f_t_trade_count_120s",
            "f_t_vol_10s","f_t_vol_30s","f_t_vol_120s",
            "f_t_burst_trade_count","f_t_burst_vol",
        ]
        out = pd.DataFrame(index=t_index, data={c: np.nan for c in cols_f})
        for c in ["f_t_trade_count_10s","f_t_trade_count_30s","f_t_trade_count_120s"]:
            out[c] = 0
        # signed/vol à 0 si pas de trades (plus stable)
        for c in ["f_t_signed_vol_10s","f_t_signed_vol_30s","f_t_signed_vol_120s",
                  "f_t_vol_10s","f_t_vol_30s","f_t_vol_120s"]:
            out[c] = 0.0
        return out

    t = tr  # pas de .copy() ici
    if not t.index.is_monotonic_increasing:
        t = t.sort_index()

    is_buy = t["is_aggr_buy"].astype(bool)
    qty = t["qty"].astype("float64")

    buy_qty = qty.where(is_buy, 0.0).resample("1s").sum()
    sell_qty = qty.where(~is_buy, 0.0).resample("1s").sum()
    tot_qty = (buy_qty + sell_qty)
    signed = (buy_qty - sell_qty)

    cnt = qty.resample("1s").count().astype("float64")

    # rolling helpers
    def rsum_float(s, w):
        return s.rolling(w, min_periods=max(1, w//3)).sum()

    def rsum_count(s, w):
        # IMPORTANT: min_periods=1 => jamais NaN au début
        return s.rolling(w, min_periods=1).sum()

    def rmean(s, w):
        return s.rolling(w, min_periods=max(1, w//3)).mean()

    def buy_ratio(w):
        b = rsum_float(buy_qty, w)
        tt = rsum_float(tot_qty, w).replace(0, np.nan)
        return (b / tt)

    feat_1s = pd.DataFrame({
        "f_t_aggr_buy_ratio_10s": buy_ratio(10),
        "f_t_aggr_buy_ratio_30s": buy_ratio(30),
        "f_t_aggr_buy_ratio_120s": buy_ratio(120),

        "f_t_signed_vol_10s": rsum_float(signed, 10),
        "f_t_signed_vol_30s": rsum_float(signed, 30),
        "f_t_signed_vol_120s": rsum_float(signed, 120),

        # COUNTS: utiliser rsum_count
        "f_t_trade_count_10s": rsum_count(cnt, 10),
        "f_t_trade_count_30s": rsum_count(cnt, 30),
        "f_t_trade_count_120s": rsum_count(cnt, 120),

        "f_t_vol_10s": rsum_float(tot_qty, 10),
        "f_t_vol_30s": rsum_float(tot_qty, 30),
        "f_t_vol_120s": rsum_float(tot_qty, 120),
    }, index=tot_qty.index)

    # burst proxies
    c10 = feat_1s["f_t_trade_count_10s"]
    v10 = feat_1s["f_t_vol_10s"]
    c120m = rmean(c10, 120).replace(0, np.nan)
    v120m = rmean(v10, 120).replace(0, np.nan)
    feat_1s["f_t_burst_trade_count"] = (c10 / c120m)
    feat_1s["f_t_burst_vol"] = (v10 / v120m)

    # align to decision timestamps
    feat_1s = feat_1s.reindex(t_index, method="ffill")

    # IMPORTANT: nettoyer avant cast
    feat_1s = feat_1s.replace([np.inf, -np.inf], np.nan)

    # counts: fillna(0) puis cast
    for c in ["f_t_trade_count_10s","f_t_trade_count_30s","f_t_trade_count_120s"]:
        feat_1s[c] = feat_1s[c].fillna(0).astype("int32")

    # volumes: fillna(0) (optionnel mais robuste)
    for c in ["f_t_signed_vol_10s","f_t_signed_vol_30s","f_t_signed_vol_120s",
              "f_t_vol_10s","f_t_vol_30s","f_t_vol_120s"]:
        feat_1s[c] = feat_1s[c].fillna(0.0).astype("float32")

    # ratios/bursts en float (on laisse NaN si pas défini)
    for c in ["f_t_aggr_buy_ratio_10s","f_t_aggr_buy_ratio_30s","f_t_aggr_buy_ratio_120s",
              "f_t_burst_trade_count","f_t_burst_vol"]:
        feat_1s[c] = feat_1s[c].astype("float32")

    return feat_1s

def cross_features(book_feat: pd.DataFrame, tr_feat: pd.DataFrame) -> pd.DataFrame:
    """
    Simple cross features (placeholders but useful):
    - flow_align_10s: sign(mid_move_10s) * sign(signed_vol_10s)
    - liquidity_stress: spread_bps * (1 + abs(signed_vol_10s))
    - impact_proxy_10s: mid_move_10s / (abs(signed_vol_10s)+eps)
    """
    eps = 1e-6
    mm = pd.to_numeric(book_feat["f_b_mid_move_10s_bps"], errors="coerce")
    sv = pd.to_numeric(tr_feat["f_t_signed_vol_10s"], errors="coerce")
    sp = pd.to_numeric(book_feat["f_b_spread_bps"], errors="coerce")

    flow_align = np.sign(mm) * np.sign(sv)
    liquidity_stress = sp * (1.0 + np.abs(sv))
    impact = mm / (np.abs(sv) + eps)

    out = pd.DataFrame({
        "f_x_flow_align_10s": flow_align.astype("float32"),
        "f_x_liquidity_stress": liquidity_stress.astype("float32"),
        "f_x_impact_proxy_10s": impact.astype("float32"),
    }, index=book_feat.index)
    return out

# =========================
# Label A (TP/SL barrier, conservative maker fill proxy)
# =========================
def first_true_idx(mask: np.ndarray) -> int:
    idx = np.flatnonzero(mask)
    return int(idx[0]) if idx.size else -1

def maker_fill_ok_mid_cross(mid_arr: np.ndarray, p0: int, fill_w: int, entry_bid: float, entry_ask: float) -> Tuple[bool,bool,int,int]:
    if p0 < 0 or p0 >= mid_arr.size:
        return False, False, -1, -1
    p1 = min(p0 + fill_w, mid_arr.size - 1)
    if p1 <= p0:
        return False, False, -1, -1
    seg = mid_arr[p0:p1+1]
    fillL = first_true_idx(seg <= entry_bid)
    fillS = first_true_idx(seg >= entry_ask)
    return (fillL >= 0), (fillS >= 0), int(fillL), int(fillS)

def maker_fill_ok_book_cross(bid_arr: np.ndarray, ask_arr: np.ndarray, p0: int, fill_w: int, entry_bid: float, entry_ask: float):
    if p0 < 0 or p0 >= bid_arr.size:
        return False, False, -1, -1
    p1 = min(p0 + fill_w, bid_arr.size - 1)
    if p1 <= p0:
        return False, False, -1, -1

    seg_bid = bid_arr[p0:p1+1]
    seg_ask = ask_arr[p0:p1+1]

    # Long: on est filled si le best ask tombe <= notre bid (prix down)
    fillL = first_true_idx(seg_ask <= entry_bid)
    # Short: filled si le best bid monte >= notre ask (prix up)
    fillS = first_true_idx(seg_bid >= entry_ask)

    return (fillL >= 0), (fillS >= 0), int(fillL), int(fillS)

def side_tp_sl_steps(seg_bid: np.ndarray, seg_ask: np.ndarray, entry_bid: float, entry_ask: float, tp_bps: float, sl_bps: float) -> Dict[str,int]:
    # LONG exits on bid; SHORT exits on ask
    tpL_px = entry_bid * (1.0 + tp_bps / 1e4)
    slL_px = entry_bid * (1.0 - sl_bps / 1e4)
    tpS_px = entry_ask * (1.0 - tp_bps / 1e4)
    slS_px = entry_ask * (1.0 + sl_bps / 1e4)

    tpL = first_true_idx(seg_bid >= tpL_px)
    slL = first_true_idx(seg_bid <= slL_px)
    tpS = first_true_idx(seg_ask <= tpS_px)
    slS = first_true_idx(seg_ask >= slS_px)
    return {"tpL": int(tpL), "slL": int(slL), "tpS": int(tpS), "slS": int(slS)}

def rr_from_costR(cost_R: np.ndarray, edges: List[float], levels: List[float]) -> np.ndarray:
    """
    edges: len K
    levels: len K+1
    """
    cost_R = np.asarray(cost_R, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.float64)
    levels = np.asarray(levels, dtype=np.float64)
    b = np.digitize(cost_R, edges, right=False)
    rr = levels[b]
    return rr.astype(np.float64)

def compute_label_A(
    t_index: pd.DatetimeIndex,
    book_1s: pd.DataFrame,
    candles_1m: pd.DataFrame,
    *,
    horizon_sec: int,
    confirm_sec: int,
    decision_step_sec: int,
    fee_exit_bps: float,
    fill_window_sec: int,
    risk_mult: float,
    risk_floor_bps: float,
    rr_costR_edges: List[float],
    rr_levels: List[float],
    confirm_thr: float,
    confirm_w_trades: float,
    confirm_w_imb: float,
    trades_feat_at_confirm: pd.DataFrame,
    book_feat_at_confirm: pd.DataFrame,
    logger: logging.Logger,
) -> pd.DataFrame:
    """
    Label A uses:
    - entry at t_confirm (t + confirm_sec) using bid/ask at that time
    - maker fill proxy in [t_confirm, t_confirm+fill_window] using mid cross
    - exit barrier check up to horizon_sec from t_confirm
    - chooses whichever side hits TP first (if any), else SL else TIME/NOFILL
    - label_A = 1 if TP happens (either side), else 0
    """

    # map decision times to confirm times
    t_confirm = (t_index + pd.to_timedelta(confirm_sec, unit="s"))

    b = book_1s  # déjà 1s, trié, ffill

    # we need book snapshots at confirm time
    entry = b.reindex(t_confirm, method="ffill")
    entry.index = t_index  # align back to decision index

    bid_entry = entry["bid0"].to_numpy(dtype="float64")
    ask_entry = entry["ask0"].to_numpy(dtype="float64")
    spread_entry = entry["spread_bps"].to_numpy(dtype="float64")
    cost_bps = 0.5 * spread_entry + float(fee_exit_bps)

    # ATR / R from candles 1m at confirm time (ffill)
    c = candles_1m.sort_index()
    # compute atr_bps
    atr = atr_ewm(c, 14)
    mid_ref = c["close"].astype("float64")
    atr_bps = (atr / mid_ref.replace(0, np.nan) * 1e4).fillna(0.0)
    atr_bps = atr_bps.reindex(t_confirm, method="ffill").to_numpy(dtype="float64")

    R_bps = np.maximum(risk_mult * atr_bps, float(risk_floor_bps)).astype(np.float64)
    denR = np.maximum(R_bps, 1e-12)
    cost_R = np.clip(cost_bps / denR, 0.0, 100.0)

    rr = rr_from_costR(cost_R, edges=rr_costR_edges, levels=rr_levels)
    tp_bps = (rr * R_bps).astype(np.float64)
    sl_bps = (R_bps).astype(np.float64)

    # EV=0 threshold diagnostic
    denom = np.maximum(tp_bps + sl_bps, 1e-12)
    p_thr_ev0 = np.clip((sl_bps + cost_bps) / denom, 0.0, 1.0).astype(np.float64)

    # CONFIRMATION RULE at t_confirm:
    # dir_score = w_trades * signed_vol_10s + w_imb * imb_L1
    # accept if abs(dir_score) >= confirm_thr
    sv10 = pd.to_numeric(trades_feat_at_confirm["f_t_signed_vol_10s"], errors="coerce").fillna(0.0).to_numpy(dtype="float64")
    imb1 = pd.to_numeric(book_feat_at_confirm["f_b_imb_L1"], errors="coerce").fillna(0.0).to_numpy(dtype="float64")

    dir_score = (confirm_w_trades * sv10) + (confirm_w_imb * imb1)
    confirm_ok = (np.abs(dir_score) >= float(confirm_thr))
    dir_side = np.where(dir_score >= 0.0, 1, -1)  # +1 LONG, -1 SHORT

    # Build time mapping into the 1s arrays for barrier checks
    b_idx = b.index
    # positions in b for each confirm time
    pos = b_idx.get_indexer(t_confirm, method="ffill").astype(np.int64)
    arr_len = len(b_idx)

    bid_arr = b["bid0"].to_numpy(dtype="float64", copy=False)
    ask_arr = b["ask0"].to_numpy(dtype="float64", copy=False)
    mid_arr = b["mid"].to_numpy(dtype="float64", copy=False)

    fill_w = max(1, int(fill_window_sec // 1))  # 1s step
    window = max(1, int(horizon_sec // 1))      # 1s step

    n = len(t_index)
    label_A = np.zeros(n, dtype=np.int8)
    exit_reason = np.full(n, "UNSET", dtype=object)
    pnl_net_bps = np.zeros(n, dtype=np.float32)
    exit_t_sec = np.full(n, -1, dtype=np.int16)
    reason = np.full(n, "unset", dtype=object)

    INF = 10**12

    for i in range(n):
        if not confirm_ok[i]:
            label_A[i] = 0
            exit_reason[i] = "CONFIRM_FAIL"
            pnl_net_bps[i] = 0.0
            exit_t_sec[i] = -1
            reason[i] = "confirm_fail"
            continue

        p0 = int(pos[i])
        if p0 < 0 or p0 >= arr_len:
            label_A[i] = 0
            exit_reason[i] = "NONE"
            pnl_net_bps[i] = 0.0
            exit_t_sec[i] = -1
            reason[i] = "no_book_pos"
            continue

        p1 = p0 + window
        if p1 >= arr_len:
            label_A[i] = 0
            exit_reason[i] = "NONE"
            pnl_net_bps[i] = 0.0
            exit_t_sec[i] = -1
            reason[i] = "not_enough_future_book"
            continue

        eL = float(bid_entry[i])
        eS = float(ask_entry[i])
        if (not np.isfinite(eL)) or (not np.isfinite(eS)) or eL <= 0 or eS <= 0:
            label_A[i] = 0
            exit_reason[i] = "NONE"
            pnl_net_bps[i] = 0.0
            exit_t_sec[i] = -1
            reason[i] = "bad_entry_px"
            continue

        okL, okS, fillL, fillS = maker_fill_ok_book_cross(bid_arr, ask_arr, p0, fill_w, eL, eS)

        # If we require direction-specific trading, only allow chosen side:
        # But Label A is "neutral", so we still label TP if ANY side could TP.
        # However, in live trading we'd choose direction from confirmation;
        # To remain consistent with the chosen direction, we compute outcome on that direction only.
        chosen = int(dir_side[i])  # +1 long, -1 short
        if chosen > 0:
            okS = False
        else:
            okL = False

        if (not okL) and (not okS):
            label_A[i] = 0
            exit_reason[i] = "NOFILL"
            pnl_net_bps[i] = 0.0
            exit_t_sec[i] = -1
            reason[i] = "nofill"
            continue

        tp_i = float(tp_bps[i])
        sl_i = float(sl_bps[i])
        c_i = float(cost_bps[i])

        tpL = slL = tpS = slS = -1

        if okL:
            p0L = p0 + fillL
            if p0L < p1:
                seg_bid = bid_arr[p0L:p1+1]
                seg_ask = ask_arr[p0L:p1+1]
                ev = side_tp_sl_steps(seg_bid, seg_ask, eL, eS, tp_i, sl_i)
                tpL, slL = ev["tpL"], ev["slL"]

        if okS:
            p0S = p0 + fillS
            if p0S < p1:
                seg_bid = bid_arr[p0S:p1+1]
                seg_ask = ask_arr[p0S:p1+1]
                ev = side_tp_sl_steps(seg_bid, seg_ask, eL, eS, tp_i, sl_i)
                tpS, slS = ev["tpS"], ev["slS"]

        def wins(tp_idx: int, sl_idx: int) -> bool:
            if tp_idx < 0:
                return False
            if sl_idx < 0:
                return True
            return tp_idx < sl_idx

        long_wins = okL and wins(tpL, slL)
        short_wins = okS and wins(tpS, slS)

        t_tp_long = (fillL + tpL) if (long_wins and tpL >= 0 and fillL >= 0) else INF
        t_tp_short = (fillS + tpS) if (short_wins and tpS >= 0 and fillS >= 0) else INF

        if long_wins or short_wins:
            label_A[i] = 1
            if t_tp_long <= t_tp_short:
                exit_reason[i] = "TP_LONG"
                exit_t_sec[i] = int(t_tp_long)
            else:
                exit_reason[i] = "TP_SHORT"
                exit_t_sec[i] = int(t_tp_short)
            pnl_net_bps[i] = np.float32(tp_i - c_i)
            reason[i] = "tp"
            continue

        # else SL if any, else TIME
        t_sl_long = (fillL + slL) if (okL and slL >= 0 and fillL >= 0) else INF
        t_sl_short = (fillS + slS) if (okS and slS >= 0 and fillS >= 0) else INF
        t_sl = min(t_sl_long, t_sl_short)

        label_A[i] = 0
        if t_sl < INF:
            if t_sl_long <= t_sl_short:
                exit_reason[i] = "SL_LONG"
                exit_t_sec[i] = int(t_sl_long)
            else:
                exit_reason[i] = "SL_SHORT"
                exit_t_sec[i] = int(t_sl_short)
            pnl_net_bps[i] = np.float32(-sl_i - c_i)
            reason[i] = "sl"
        else:
            exit_reason[i] = "TIME"
            exit_t_sec[i] = -1
            pnl_net_bps[i] = 0.0
            reason[i] = "time"

    out = pd.DataFrame({
    "label_A": label_A.astype("int8"),
    "label_A_reason": pd.Series(reason, index=t_index, dtype="string"),
    "label_A_exit_reason": pd.Series(exit_reason, index=t_index, dtype="string"),
    "label_A_pnl_net_bps": pnl_net_bps.astype("float32"),
    "label_A_exit_t_sec": exit_t_sec.astype("int16"),
    "audit_spread_bps_entry": spread_entry.astype("float32"),
    "audit_cost_bps": cost_bps.astype("float32"),
    "audit_R_bps": R_bps.astype("float32"),
    "audit_tp_bps": tp_bps.astype("float32"),
    "audit_sl_bps": sl_bps.astype("float32"),
    "audit_rr_min": rr.astype("float32"),
    "audit_cost_R": cost_R.astype("float32"),
    "audit_p_thr_ev0": p_thr_ev0.astype("float32"),
}, index=t_index)

    return out

# =========================
# Main stageA builder per month
# =========================
@dataclass
class StageAConfig:
    symbol: str
    year: int
    month: int
    book_root: str
    trade_root: str
    candle_root: str
    out_root: str
    aws_region: Optional[str]
    horizon_sec: int
    decision_step_sec: int
    confirm_sec: int
    fill_window_sec: int
    fee_exit_bps: float
    risk_mult: float
    risk_floor_bps: float
    rr_costR_edges: List[float]
    rr_levels: List[float]
    confirm_thr: float
    confirm_w_trades: float
    confirm_w_imb: float

def load_candles_1m(path: str, so: dict, logger: logging.Logger) -> pd.DataFrame:
    if not exists(path, so):
        raise FileNotFoundError(f"Missing candle file: {path}")
    c = pd.read_parquet(path, storage_options=so)
    c["timestamp"] = ensure_utc(c["timestamp"])
    c = c.dropna(subset=["timestamp"]).sort_values("timestamp").set_index("timestamp", drop=True)
    # basic OHLC sanity
    c = c.replace([np.inf, -np.inf], np.nan).dropna(subset=["open","high","low","close"])
    return c

def book_needed_columns(df_cols: List[str], L: int = 15) -> List[str]:
    cols = ["timestamp"]

    # prix top of book
    if "bid_0_price" in df_cols and "ask_0_price" in df_cols:
        cols += ["bid_0_price", "ask_0_price"]
    elif "bid0" in df_cols and "ask0" in df_cols:
        cols += ["bid0", "ask0"]
    elif "bid" in df_cols and "ask" in df_cols:
        cols += ["bid", "ask"]

    # tailles
    for i in range(L):
        if f"bid_{i}_size" in df_cols: cols.append(f"bid_{i}_size")
        if f"ask_{i}_size" in df_cols: cols.append(f"ask_{i}_size")
        if f"bid_{i}_qty" in df_cols:  cols.append(f"bid_{i}_qty")
        if f"ask_{i}_qty" in df_cols:  cols.append(f"ask_{i}_qty")

    # uniq
    return list(dict.fromkeys(cols))

def load_book_month(path: str, so: dict, logger: logging.Logger) -> pd.DataFrame:
    if not exists(path, so):
        raise FileNotFoundError(f"Missing book month file: {path}")

    # lire schema sans tout charger
    with fsspec.open(path, "rb", **so) as f:
        pf = pq.ParquetFile(f)
        df_cols = pf.schema.names

    cols = book_needed_columns(df_cols, L=15)

    df = pd.read_parquet(path, storage_options=so, columns=cols, engine="pyarrow")
    b = normalize_book_l15(df)
    b = b.sort_index()
    return b

def load_trades_month(path: str, so: dict, logger: logging.Logger) -> pd.DataFrame:
    if not exists(path, so):
        log(logger, f"[WARN] trades file missing: {path} -> empty trades")
        return pd.DataFrame()
    df = pd.read_parquet(path, storage_options=so)
    t = normalize_trades(df)
    return t

def build_decision_index(book: pd.DataFrame, year: int, month: int, step_sec: int) -> pd.DatetimeIndex:
    # Restrict to month span
    start = pd.Timestamp(year=year, month=month, day=1, tz="UTC")
    if month == 12:
        end = pd.Timestamp(year=year+1, month=1, day=1, tz="UTC") - pd.Timedelta(seconds=1)
    else:
        end = pd.Timestamp(year=year, month=month+1, day=1, tz="UTC") - pd.Timedelta(seconds=1)

    # Use book span intersection (avoid creating timestamps outside available data)
    bmin = book.index.min()
    bmax = book.index.max()
    s = max(start, bmin.floor("s"))
    e = min(end, bmax.floor("s"))

    if e <= s:
        return pd.DatetimeIndex([], tz="UTC")

    # align to step_sec grid
    freq = f"{int(step_sec)}s"
    idx = pd.date_range(start=s.ceil(freq), end=e.floor(freq), freq=freq, tz="UTC")
    return idx

def build_stageA_month(cfg: StageAConfig, logger: logging.Logger):
    so = s3_so(cfg.aws_region)

    ym = yyyy_mm(cfg.year, cfg.month)

    book_path = f"{cfg.book_root.rstrip('/')}/{cfg.symbol}/{ym}.parquet"
    trade_path = f"{cfg.trade_root.rstrip('/')}/{cfg.symbol}/{ym}.parquet"
    candle_path = f"{cfg.candle_root.rstrip('/')}/{cfg.symbol}/{cfg.symbol}-1m-{cfg.year:04d}.parquet"

    log(logger, f"StageA {cfg.symbol} {ym}: loading book={book_path}")
    book = load_book_month(book_path, so, logger)
    if book.empty:
        log(logger, f"[WARN] empty book for {cfg.symbol} {ym} -> skip")
        return
    
    size_cols = [c for c in book.columns if (c.startswith("bid_") or c.startswith("ask_")) and (c.endswith("_size") or c.endswith("_qty"))]
    log(logger, f"Book premium cols: sizes_found={len(size_cols)} examples={size_cols[:6]}")
    if len(size_cols) == 0:
        log(logger, "[WARN] No L15 size/qty columns found in book -> premium features will be NaN")

    log(logger, f"StageA {cfg.symbol} {ym}: loading trades={trade_path}")
    trades = load_trades_month(trade_path, so, logger)

    log(logger, f"StageA {cfg.symbol} {cfg.year}: loading candles={candle_path}")
    candles = load_candles_1m(candle_path, so, logger)

    # decision grid
    t_idx = build_decision_index(book, cfg.year, cfg.month, cfg.decision_step_sec)
    if len(t_idx) == 0:
        log(logger, f"[WARN] no decision timestamps in month {ym} -> skip")
        return
    log(logger, f"Decision grid: n={len(t_idx):,} step={cfg.decision_step_sec}s span={t_idx.min().isoformat()} → {t_idx.max().isoformat()}")

    def restrict_time(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        if df is None or df.empty:
            return df
        return df.loc[(df.index >= start) & (df.index <= end)]

    lookback_sec = 180
    future_sec = cfg.confirm_sec + cfg.horizon_sec + cfg.fill_window_sec + 10

    win_start = t_idx.min() - pd.Timedelta(seconds=lookback_sec)
    win_end   = t_idx.max() + pd.Timedelta(seconds=future_sec)

    book   = restrict_time(book, win_start, win_end)
    trades = restrict_time(trades, win_start, win_end)
    candles = restrict_time(candles, win_start.floor("1min"), win_end.ceil("1min"))

    # book, trades, candles are restricted to [win_start, win_end]

    # Build book_1s once for label
    log(logger, "Build book_1s base...")
    
    # garder toutes les colonnes (prix + tailles) et resampler 1s
    book_1s = book.resample("1s").last().ffill()
    
    # après book_1s = book.resample("1s").last().ffill()
    # cast uniquement sur les colonnes vraiment numériques
    num_cols = book_1s.select_dtypes(include=["number"]).columns
    book_1s[num_cols] = book_1s[num_cols].astype("float32")


    # SUPER IMPORTANT: on n'a plus besoin du book raw ensuite (énorme)
    del book
    gc.collect()

    # Output partition dir
    part_dir = out_partition(cfg.out_root, cfg.symbol, cfg.year, cfg.month)
    safe_mkdir(part_dir, so)

    # Chunk params
    chunk_size = 20_000  # tune (25k if still tight)
    n_total = len(t_idx)
    n_parts = (n_total + chunk_size - 1) // chunk_size

    log(logger, f"Chunked build: n={n_total:,} chunk_size={chunk_size:,} parts={n_parts}")

    tp_sum = sl_sum = nf_sum = cf_sum = 0

    for part_id in range(n_parts):
        s0 = part_id * chunk_size
        s1 = min(n_total, (part_id + 1) * chunk_size)
        t_part = t_idx[s0:s1]

        log(logger, f"[{ym}] part {part_id+1}/{n_parts}: rows {s0:,}-{s1-1:,} (n={len(t_part):,})")

        # --- features on this chunk only ---
        feat_c = candle_features_1m(candles, t_part)
        feat_b = book_depth_features(book_1s, t_part, L=15)
        feat_t = trades_features(trades, t_part)
        feat_x = cross_features(feat_b, feat_t)

        nn = feat_b["f_b_imb_L1"].notna().mean()
        log(logger, f"[{ym}] part {part_id:05d} premium_check: f_b_imb_L1 non-null ratio={nn:.3f}")

        # confirmation snapshots (t+confirm)
        t_confirm_part = t_part + pd.to_timedelta(cfg.confirm_sec, unit="s")
        feat_b_confirm = book_depth_features(book_1s, t_confirm_part, L=15)
        feat_b_confirm.index = t_part
        feat_t_confirm = trades_features(trades, t_confirm_part)
        feat_t_confirm.index = t_part

        # --- label on this chunk only ---
        lab = compute_label_A(
            t_index=t_part,
            book_1s=book_1s,
            candles_1m=candles,
            horizon_sec=cfg.horizon_sec,
            confirm_sec=cfg.confirm_sec,
            decision_step_sec=cfg.decision_step_sec,
            fee_exit_bps=cfg.fee_exit_bps,
            fill_window_sec=cfg.fill_window_sec,
            risk_mult=cfg.risk_mult,
            risk_floor_bps=cfg.risk_floor_bps,
            rr_costR_edges=cfg.rr_costR_edges,
            rr_levels=cfg.rr_levels,
            confirm_thr=cfg.confirm_thr,
            confirm_w_trades=cfg.confirm_w_trades,
            confirm_w_imb=cfg.confirm_w_imb,
            trades_feat_at_confirm=feat_t_confirm,
            book_feat_at_confirm=feat_b_confirm,
            logger=logger,
        )

        log(logger, f"[{ym}] part {part_id:05d} label_exit_reason NA ratio={lab['label_A_exit_reason'].isna().mean():.3f}")
        
        # --- ids + cfg/meta for chunk ---
        out = pd.DataFrame(index=t_part)
        out["id_symbol"] = cfg.symbol
        out["id_t"] = t_part
        out["id_year"] = np.int16(cfg.year)
        out["id_month"] = np.int8(cfg.month)
        out["id_row_id"] = [row_id(cfg.symbol, pd.Timestamp(t).isoformat().replace("+00:00","Z"), salt="stageA") for t in t_part]

        out["cfg_horizon_sec"] = np.int16(cfg.horizon_sec)
        out["cfg_decision_step_sec"] = np.int16(cfg.decision_step_sec)
        out["cfg_k_confirm_sec"] = np.int16(cfg.confirm_sec)
        out["cfg_fee_exit_bps"] = np.float32(cfg.fee_exit_bps)
        out["cfg_cost_model"] = "half_spread_plus_fee_exit"

        out = pd.concat([out, lab, feat_c, feat_b, feat_t, feat_x], axis=1)

        # reorder columns
        try:
            schema = build_stageA_schema()
            expected = [c["name"] for c in schema.get("all_columns_ordered", [])]
            present = [c for c in expected if c in out.columns]
            extras = [c for c in out.columns if c not in set(present)]
            out = out[present + extras]
        except Exception:
            pass

        # write part parquet
        out_path = f"{part_dir}/part-{part_id:05d}.parquet"
        log(logger, f"Write -> {out_path} (rows={len(out):,} cols={out.shape[1]})")
        write_parquet(out_path, out.reset_index(drop=True), so, compression="snappy")

        vc = out["label_A_exit_reason"].astype("string").value_counts(dropna=False).head(20)
        log(logger, f"[{ym}] part {part_id:05d} exit_reason top:\n{vc.to_string()}")
        log(logger, f"[{ym}] part {part_id:05d} label_A mean={out['label_A'].mean():.4f}")

        # stats per part
        er = out["label_A_exit_reason"].astype("string").fillna("")

        tp = int((er == "TP_LONG").sum() + (er == "TP_SHORT").sum())
        sl = int((er == "SL_LONG").sum() + (er == "SL_SHORT").sum())
        nf = int((er == "NOFILL").sum())
        cf = int((er == "CONFIRM_FAIL").sum())      
        
        tp_sum += tp; sl_sum += sl; nf_sum += nf; cf_sum += cf

        # cleanup per chunk
        del feat_c, feat_b, feat_t, feat_x, feat_b_confirm, feat_t_confirm, lab, out
        gc.collect()

    log(logger, f"Done {cfg.symbol} {ym}: TP={tp_sum} SL={sl_sum} NOFILL={nf_sum} CONFIRM_FAIL={cf_sum}")

    # cleanup month
    del trades, candles, book_1s
    gc.collect()
    return

# =========================
# CLI
# =========================
def parse_float_list(s: str) -> List[float]:
    return [float(x) for x in str(s).split(",") if str(x).strip() != ""]

def main():
    ap = argparse.ArgumentParser("Build StageA dataset (BTC 2m horizon, 30s decisions, confirm at +20s).")

    ap.add_argument("--symbols", nargs="+", default=["BTCUSDT"])
    ap.add_argument("--years", nargs="+", type=int, default=[2024])
    ap.add_argument("--months", nargs="+", type=int, default=None)

    ap.add_argument("--book-root", default="s3://tradebot-config-tokyo/data/book")
    ap.add_argument("--trade-root", default="s3://tradebot-config-tokyo/data/trade")
    ap.add_argument("--candle-root", default="s3://tradebot-config-tokyo/data/bougie")
    ap.add_argument("--out-root", default="s3://tradebot-config-tokyo/data/stageA")
    ap.add_argument("--aws-region", default="ap-northeast-1")

    # timing
    ap.add_argument("--horizon-sec", type=int, default=120)
    ap.add_argument("--decision-step-sec", type=int, default=30)
    ap.add_argument("--confirm-sec", type=int, default=20)
    ap.add_argument("--fill-window-sec", type=int, default=10)

    # cost/risk
    ap.add_argument("--fee-exit-bps", type=float, default=6.0)
    ap.add_argument("--risk-mult", type=float, default=1.0)
    ap.add_argument("--risk-floor-bps", type=float, default=8.0)

    # RR design A via cost_R bins (defaults aligned with your earlier setup)
    ap.add_argument("--rr-costR-edges", type=str, default="0.35,0.60,0.90")
    ap.add_argument("--rr-levels", type=str, default="2.2,2.6,3.0,3.0")  # len = len(edges)+1

    # Confirmation rule params
    ap.add_argument("--confirm-thr", type=float, default=50.0, help="threshold on |dir_score| to accept trade")
    ap.add_argument("--confirm-w-trades", type=float, default=1.0, help="weight on signed trade volume 10s")
    ap.add_argument("--confirm-w-imb", type=float, default=100.0, help="weight on book imbalance L1 (scaled)")

    ap.add_argument("--write-schema", action="store_true", help="write schemaA.json to out-root and exit")
    ap.add_argument("--verbose", action="store_true")

    args = ap.parse_args()
    logger = setup_logger(verbose=args.verbose)

    so = s3_so(args.aws_region)
    out_root = args.out_root.rstrip("/")
    schema_path = f"{out_root}/schemaA.json"

    # write schema once (or if requested)
    if (args.write_schema) or (not exists(schema_path, so)):
        schema = build_stageA_schema()
        log(logger, f"Writing schema -> {schema_path}")
        write_json(schema_path, schema, so)
        if args.write_schema:
            log(logger, "Done (schema only).")
            return

    edges = parse_float_list(args.rr_costR_edges)
    levels = parse_float_list(args.rr_levels)
    if len(levels) != len(edges) + 1:
        raise ValueError(f"--rr-levels must have len(edges)+1. got edges={len(edges)} levels={len(levels)}")

    # Process months
    for sym in args.symbols:
        for (y, m) in month_iter(args.years, args.months):
            cfg = StageAConfig(
                symbol=sym,
                year=int(y),
                month=int(m),
                book_root=str(args.book_root),
                trade_root=str(args.trade_root),
                candle_root=str(args.candle_root),
                out_root=str(args.out_root),
                aws_region=str(args.aws_region),
                horizon_sec=int(args.horizon_sec),
                decision_step_sec=int(args.decision_step_sec),
                confirm_sec=int(args.confirm_sec),
                fill_window_sec=int(args.fill_window_sec),
                fee_exit_bps=float(args.fee_exit_bps),
                risk_mult=float(args.risk_mult),
                risk_floor_bps=float(args.risk_floor_bps),
                rr_costR_edges=edges,
                rr_levels=levels,
                confirm_thr=float(args.confirm_thr),
                confirm_w_trades=float(args.confirm_w_trades),
                confirm_w_imb=float(args.confirm_w_imb),
            )
            try:
                build_stageA_month(cfg, logger)
            except FileNotFoundError as e:
                log(logger, f"[WARN] {e} -> skip {sym} {yyyy_mm(y,m)}")
            except Exception as e:
                log(logger, f"[ERROR] failed {sym} {yyyy_mm(y,m)}: {type(e).__name__}: {e}")
                raise

if __name__ == "__main__":
    main()