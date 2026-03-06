from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple


def bootstrap_iid(x: np.ndarray, n_boot: int, seed: int) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(x)
    if n == 0:
        return (np.nan, np.nan)
    boots = []
    for _ in range(n_boot):
        samp = x[rng.integers(0, n, size=n)]
        boots.append(float(np.mean(samp)))
    return (float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975)))

def bootstrap_block_delta_by_day(df_a: pd.DataFrame, df_b: pd.DataFrame, *,
                                 value_col: str, day_col: str,
                                 n_boot: int, seed: int = 7) -> tuple[float, float]:
    """
    Returns CI95 for (mean(df_b) - mean(df_a)) with block bootstrap by day.
    Each side is resampled by day independently (conservative, avoids assuming paired events).
    """
    rng = np.random.default_rng(int(seed))

    a = df_a.dropna(subset=[value_col, day_col]).copy()
    b = df_b.dropna(subset=[value_col, day_col]).copy()

    days_a = a[day_col].drop_duplicates().to_numpy()
    days_b = b[day_col].drop_duplicates().to_numpy()

    if len(days_a) == 0 or len(days_b) == 0:
        return (np.nan, np.nan)

    # pre-group for speed
    grp_a = {d: a.loc[a[day_col] == d, value_col].to_numpy(dtype="float64") for d in days_a}
    grp_b = {d: b.loc[b[day_col] == d, value_col].to_numpy(dtype="float64") for d in days_b}

    deltas = np.empty(int(n_boot), dtype="float64")

    for i in range(int(n_boot)):
        samp_a = rng.choice(days_a, size=len(days_a), replace=True)
        samp_b = rng.choice(days_b, size=len(days_b), replace=True)

        xa = np.concatenate([grp_a[d] for d in samp_a]) if len(samp_a) else np.array([], dtype="float64")
        xb = np.concatenate([grp_b[d] for d in samp_b]) if len(samp_b) else np.array([], dtype="float64")

        ma = float(np.mean(xa)) if len(xa) else np.nan
        mb = float(np.mean(xb)) if len(xb) else np.nan
        deltas[i] = (mb - ma) if np.isfinite(ma) and np.isfinite(mb) else np.nan

    deltas = deltas[np.isfinite(deltas)]
    if len(deltas) == 0:
        return (np.nan, np.nan)

    lo = float(np.quantile(deltas, 0.025))
    hi = float(np.quantile(deltas, 0.975))
    return (lo, hi)

def bootstrap_block_by_day(df: pd.DataFrame, value_col: str, day_col: str,
                           n_boot: int, seed: int) -> Tuple[float, float]:
    """
    Resample days with replacement, concatenate trades of sampled days.
    """
    rng = np.random.default_rng(seed)
    g = df.groupby(day_col)[value_col].apply(lambda s: s.dropna().to_numpy())
    days = g.index.to_numpy()
    blocks = g.to_list()
    m = len(days)
    if m == 0:
        return (np.nan, np.nan)

    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, m, size=m)
        # concat blocks
        samp = np.concatenate([blocks[i] for i in idx if len(blocks[i]) > 0], axis=0)
        boots.append(float(np.mean(samp)) if len(samp) else np.nan)

    boots = np.array([b for b in boots if np.isfinite(b)], dtype="float64")
    if len(boots) == 0:
        return (np.nan, np.nan)
    return (float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975)))