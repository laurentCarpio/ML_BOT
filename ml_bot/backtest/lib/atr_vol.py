from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple, List

from .io_s3 import read_parquet_s3
from .s3_paths import candles_year_path


def compute_atr_bps_from_1m(candles_1m: pd.DataFrame, atr_n: int, tf: str) -> pd.DataFrame:
    """
    candles_1m: timestamp, open, high, low, close, volume
    returns df with timestamp (tf bars end), atr_bps
    """
    c = candles_1m.copy()
    c["timestamp"] = pd.to_datetime(c["timestamp"], utc=True)
    c = c.sort_values("timestamp")
    for col in ["open", "high", "low", "close"]:
        c[col] = pd.to_numeric(c[col], errors="coerce")
    c = c.dropna(subset=["timestamp", "high", "low", "close"])

    # resample to tf (OHLC)
    c = c.set_index("timestamp")
    ohlc = c.resample(tf).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum"
    }).dropna(subset=["open", "high", "low", "close"]).reset_index()

    prev_close = ohlc["close"].shift(1)
    tr = np.maximum(
        ohlc["high"] - ohlc["low"],
        np.maximum((ohlc["high"] - prev_close).abs(), (ohlc["low"] - prev_close).abs())
    )
    atr = tr.rolling(atr_n, min_periods=atr_n).mean()
    ohlc["atr"] = atr
    ohlc["atr_bps"] = (ohlc["atr"] / ohlc["close"]) * 1e4
    return ohlc[["timestamp", "atr_bps"]]


def attach_vol_bucket(trades: pd.DataFrame, atr_tf_df: pd.DataFrame) -> pd.DataFrame:
    """
    trades: timestamp (trade time), pnl_net_bps, config
    atr_tf_df: timestamp (tf bar), atr_bps
    attaches nearest past atr (asof backward), then tercile bucket.
    """
    t = trades.copy()
    t["timestamp"] = pd.to_datetime(t["timestamp"], utc=True)
    t = t.sort_values("timestamp")
    a = atr_tf_df.copy().sort_values("timestamp")

    t2 = pd.merge_asof(t, a, on="timestamp", direction="backward")
    t2 = t2.dropna(subset=["atr_bps"]).copy()

    q1 = float(t2["atr_bps"].quantile(1/3))
    q2 = float(t2["atr_bps"].quantile(2/3))

    def bucket(v):
        if v <= q1:
            return "low_vol"
        if v <= q2:
            return "mid_vol"
        return "high_vol"

    t2["vol_bucket"] = t2["atr_bps"].map(bucket)
    t2["t_atr"] = t2["timestamp"]  # keep column name used in notebook
    return t2


def load_candles_years(candles_root: str, symbol: str, years: List[int]) -> pd.DataFrame:
    parts = []
    for y in years:
        p = candles_year_path(candles_root, symbol, int(y))
        df = read_parquet_s3(p)
        parts.append(df)
    c = pd.concat(parts, ignore_index=True).sort_values("timestamp")
    return c