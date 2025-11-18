# anomaly_scan.py
from typing import Dict, Any
import pandas as pd
import numpy as np

def scan_anomalies(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    rules = cfg["anomaly_scan"]["rules"]
    sev = cfg["anomaly_scan"]["severity_weights"]

    events = []

    # 1) Flat candles
    close = df["close"].values
    flat = (pd.Series(close).diff().fillna(0) == 0)
    flat_runs = (flat.groupby((~flat).cumsum()).transform('sum')) * flat
    idx_flat = df.index[flat_runs >= rules["flat_candle_min"]]
    for i in idx_flat:
        events.append({"timestamp": df.at[i,"timestamp"], "type":"flat_candle","severity":sev["flat_candle"]})

    # 2) Zero volume runs
    vol = df["volume"].values
    zero = (vol == 0)
    zero_runs = (pd.Series(zero).groupby((~pd.Series(zero)).cumsum()).transform('sum')) * zero
    idx_zero = df.index[zero_runs >= rules["zero_volume_min"]]
    for i in idx_zero:
        events.append({"timestamp": df.at[i,"timestamp"], "type":"zero_volume","severity":sev["zero_volume"]})

    # 3) Big jumps close-to-close
    pct = pd.Series(close).pct_change()
    jumps = pct.abs() > rules["pct_jump_close"]
    for i in df.index[jumps.fillna(False)]:
        events.append({"timestamp": df.at[i,"timestamp"], "type":"pct_jump","severity":sev["pct_jump"]})

    # 4) OHLC guards
    if rules.get("high_low_guard", True):
        bad_h = df["high"] < df[["open","close"]].max(axis=1)
        bad_l = df["low"]  > df[["open","close"]].min(axis=1)
        for i in df.index[bad_h | bad_l]:
            events.append({"timestamp": df.at[i,"timestamp"], "type":"ohlc_inconsistent","severity":sev["ohlc_inconsistent"]})

    # 5) Negative/NaN guards
    if rules.get("neg_values_guard", True):
        bad = (df[["open","high","low","close","volume"]] < 0).any(axis=1) | (~np.isfinite(df[["open","high","low","close","volume"]])).any(axis=1)
        for i in df.index[bad]:
            events.append({"timestamp": df.at[i,"timestamp"], "type":"negative_values","severity":sev["negative_values"]})

    out = pd.DataFrame(events).sort_values("timestamp")
    return out