from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import List, Tuple
import pandas as pd


def months_between(start_ym: str, end_ym: str) -> List[str]:
    """Inclusive months list between YYYY-MM and YYYY-MM."""
    a = pd.Period(start_ym, freq="M")
    b = pd.Period(end_ym, freq="M")
    if b < a:
        raise ValueError("end < start")
    return [str(p) for p in pd.period_range(a, b, freq="M")]


def next_month_str(ym: str) -> str:
    p = pd.Period(ym, freq="M")
    return str(p + 1)


def utc_ts(x) -> pd.Timestamp:
    return pd.to_datetime(x, utc=True)
