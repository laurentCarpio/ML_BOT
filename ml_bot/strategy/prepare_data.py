# next_bot/strategy/prepare_data.py
from __future__ import annotations
import pandas as pd
import numpy as np

# ⚙️ Garde-fous (mêmes valeurs que dans la stratégie)
SLOPE_MIN    = 5e-4    # ~0.05%/barre
EXT_MAX_ATR  = 1.2     # extension max vs SMA20 au signal
TOUCH_BAND   = 0.20    # tolérance de "retour vers 20" (pondéré par ATR/close)
ATR_LEN      = 14
SMA_FAST     = 20
SMA_SLOW     = 200
PIVOT_LEN    = 3       # minibase

def _sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()

def _atr(df: pd.DataFrame, n: int = ATR_LEN) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    # Wilder smoothing (EMA alpha=1/n)
    return tr.ewm(alpha=1/n, adjust=False).mean()

def calculate_indicators(self, df: pd.DataFrame, frequency: str) -> pd.DataFrame:
    """
    Enrichit df (OHLCV) avec SMA20/200, ATR14, pente SMA20, extension vs 20,
    et mini-pivots (pour la reprise).
    """
    if df is None or df.empty:
        return df

    out = df.copy()

    # Tri temporel (sécurité)
    if not out["timestamp"].is_monotonic_increasing:
        out = out.sort_values("timestamp", kind="stable", ignore_index=True)

    # Indicateurs de base
    out["sma20"]   = _sma(out["close"], SMA_FAST)
    out["sma200"]  = _sma(out["close"], SMA_SLOW)
    out["atr14"]   = _atr(out, ATR_LEN)

    # Pente normalisée de la SMA20 (~%/barre)
    out["sma20_slope"] = (out["sma20"] - out["sma20"].shift(5)) / (out["close"] * 5)

    # Distance normalisée (extension) vs sma20
    out["dist20_atr"] = (out["close"] - out["sma20"]).abs() / out["atr14"]

    # "Touch" zone: a touch/retest de la 20 avec tolérance
    # On ne met pas de flag ici; on calculera la condition dans la stratégie (pulled_back)

    # Mini-pivots (pour la reprise au-dessus/au-dessous)
    out["pivot_high"] = out["high"].rolling(PIVOT_LEN, min_periods=PIVOT_LEN).max().shift(1)
    out["pivot_low"]  = out["low"].rolling(PIVOT_LEN,  min_periods=PIVOT_LEN).min().shift(1)

    # Downcast utile
    for c in ("open","high","low","close","volume","sma20","sma200","atr14",
              "sma20_slope","dist20_atr","pivot_high","pivot_low"):
        if c in out.columns:
            out[c] = out[c].astype("float32")

    return out


# =========================
# Retail A/B/C/D indicators
# =========================

def sma(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n, min_periods=n).mean()

def bbands(df: pd.DataFrame, n: int = 20, k: float = 2.0):
    mid = sma(df["close"], n)
    std = df["close"].rolling(n, min_periods=n).std()
    upper = mid + k * std
    lower = mid - k * std
    return upper, mid, lower

def bb_width(df: pd.DataFrame, n: int = 20, k: float = 2.0) -> pd.Series:
    upper, mid, lower = bbands(df, n=n, k=k)
    width = (upper - lower) / mid
    return width

def rsi(df: pd.DataFrame, n: int = 14) -> pd.Series:
    delta = df["close"].diff()
    up = (delta.clip(lower=0)).rolling(n, min_periods=n).mean()
    down = (-delta.clip(upper=0)).rolling(n, min_periods=n).mean()
    rs = up / (down.replace(0, np.nan))
    return 100 - (100 / (1 + rs))

def add_retail_indicators(df: pd.DataFrame,
                          bb_n: int = 20,
                          bb_k: float = 2.0,
                          rsi_n: int = 14) -> pd.DataFrame:
    """Add minimal indicators required by retail A/B/C/D:
        - sma20, sma200
        - bb_width
        - rsi (period rsi_n)
    Leaves existing columns untouched; computes only if missing.
    """
    if df is None or df.empty:
        return df
    out = df.copy()
    if "sma20" not in out.columns:
        out["sma20"] = sma(out["close"], 20)
    if "sma200" not in out.columns:
        out["sma200"] = sma(out["close"], 200)
    if "bb_width" not in out.columns:
        out["bb_width"] = bb_width(out, n=bb_n, k=bb_k)
    if "rsi" not in out.columns:
        out["rsi"] = rsi(out, n=rsi_n)

    # light downcast
    for c in ("sma20","sma200","bb_width","rsi"):
        if c in out.columns:
            out[c] = out[c].astype("float32")

    return out

# --- Optional hook: if calculate_indicators exists and returns 'out', ensure retail features are present
try:
    # wrap original calculate_indicators if present
    _orig_calculate_indicators = calculate_indicators
    def calculate_indicators(*args, **kwargs):
        out = _orig_calculate_indicators(*args, **kwargs)
        try:
            return add_retail_indicators(out)
        except Exception:
            return out
except NameError:
    pass
