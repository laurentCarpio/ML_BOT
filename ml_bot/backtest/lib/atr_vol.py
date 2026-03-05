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

def attach_vol_bucket(trades: pd.DataFrame, atr_tf_df: pd.DataFrame, n_buckets: int = 3) -> pd.DataFrame:
    """
    trades: timestamp (trade time), pnl_net_bps, config
    atr_tf_df: timestamp (tf bar), atr_bps
    attaches nearest past atr (asof backward), then quantile buckets.
    n_buckets=3 -> low/mid/high
    n_buckets=5 -> b0..b4 (or keep names if you prefer)
    """
    t = trades.copy()
    t["timestamp"] = pd.to_datetime(t["timestamp"], utc=True)
    t = t.sort_values("timestamp")
    a = atr_tf_df.copy().sort_values("timestamp")

    t2 = pd.merge_asof(t, a, on="timestamp", direction="backward")
    t2 = t2.dropna(subset=["atr_bps"]).copy()

    nb = int(n_buckets)
    if nb < 2:
        raise ValueError("n_buckets must be >= 2")

    # quantile edges: 0, 1/nb, 2/nb, ..., 1
    qs = [i / nb for i in range(1, nb)]
    edges = [float(t2["atr_bps"].quantile(q)) for q in qs]

    def bucket(v: float) -> str:
        # bucket index in [0..nb-1]
        k = 0
        for e in edges:
            if v <= e:
                return f"b{k}"
            k += 1
        return f"b{nb-1}"

    t2["vol_bucket"] = t2["atr_bps"].map(bucket)

    # optional: keep legacy names when nb==3
    if nb == 3:
        t2["vol_bucket"] = t2["vol_bucket"].map({"b0": "low_vol", "b1": "mid_vol", "b2": "high_vol"})

    t2["t_atr"] = t2["timestamp"]
    return t2

def load_candles_years(candles_root: str, symbol: str, years: List[int]) -> pd.DataFrame:
    parts = []
    for y in years:
        p = candles_year_path(candles_root, symbol, int(y))
        df = read_parquet_s3(p)
        parts.append(df)
    c = pd.concat(parts, ignore_index=True).sort_values("timestamp")
    return c