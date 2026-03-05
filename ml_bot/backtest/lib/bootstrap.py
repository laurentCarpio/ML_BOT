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