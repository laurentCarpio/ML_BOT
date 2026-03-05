from __future__ import annotations
import numpy as np
import pandas as pd
import math
from typing import Tuple


def build_windows(events_ts: pd.Series, pre_s: float, post_s: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns merged non-overlapping windows [start,end] as datetime64[ns] arrays.
    """
    t = pd.to_datetime(events_ts, utc=True).sort_values()
    t = pd.to_datetime(t).values.astype("datetime64[ns]")

    pre  = np.timedelta64(int(math.ceil(pre_s  * 1e9)), "ns")
    post = np.timedelta64(int(math.ceil(post_s * 1e9)), "ns")

    starts = t - pre
    ends = t + post

    if len(starts) == 0:
        return np.array([], dtype="datetime64[ns]"), np.array([], dtype="datetime64[ns]")

    merged_s = []
    merged_e = []
    cur_s, cur_e = starts[0], ends[0]
    for s, e in zip(starts[1:], ends[1:]):
        if s <= cur_e:
            if e > cur_e:
                cur_e = e
        else:
            merged_s.append(cur_s)
            merged_e.append(cur_e)
            cur_s, cur_e = s, e
    merged_s.append(cur_s)
    merged_e.append(cur_e)

    return np.array(merged_s, dtype="datetime64[ns]"), np.array(merged_e, dtype="datetime64[ns]")


def overlaps_any(a_min: np.datetime64, a_max: np.datetime64, win_s: np.ndarray, win_e: np.ndarray) -> bool:
    """
    windows must be sorted & non-overlapping.
    """
    if len(win_s) == 0:
        return False
    i = np.searchsorted(win_e, a_min, side="left")
    if i >= len(win_s):
        return False
    return win_s[i] <= a_max