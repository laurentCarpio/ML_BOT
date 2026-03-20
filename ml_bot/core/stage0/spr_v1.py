# ml_bot/core/stage0/spr_v1.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable, Tuple, Dict, Any, Optional
from collections import OrderedDict

import numpy as np
import pandas as pd


# =============================================================================
# Config
# =============================================================================

@dataclass(frozen=True)
class Stage0SPRConfig:
    trades_window_s: float = 2.0
    persist_window_ms: int = 800
    thinning_window_s: float = 2.0

    profit_net_target_bps: float = 6.0
    fee_maker_bps: float = 2.0
    fee_taker_bps: float = 6.0
    slip_bps: float = 2.0

    # thresholds percentiles (fit on train)
    p_spread: float = 60.0
    p_micro: float = 80.0
    p_obi10: float = 97.0
    p_ti: float = 85.0
    p_nps: float = 60.0
    p_thin: float = 85.0
    p_range60s: float = 90.0  # percentile for range_60s_bps MIN

    persist_ms_min: int = 250
    ms_keep_quantile: float = 97.0

    depths: Tuple[int, ...] = (1, 3, 5, 10, 14)
    max_horizon_s: float = 300.0

    # spread/liquidity features
    tick_size: float = 0.1
    spread_roll_med_s: int = 300
    p_spread_rel: float = 80.0      # percentile for spread_rel_5m MAX
    p_spread_ticks: float = 80.0    # percentile for spread_ticks_1s MAX

    @property
    def fees_rt_bps(self) -> float:
        return float(self.fee_maker_bps + self.fee_taker_bps)

    @property
    def thr_exec_bps(self) -> float:
        return float(self.profit_net_target_bps + self.fees_rt_bps + self.slip_bps)

    @staticmethod
    def parse_depths(s: str) -> Tuple[int, ...]:
        parts = [p.strip() for p in str(s).split(",") if p.strip()]
        out = tuple(sorted(set(int(x) for x in parts)))
        if not out:
            raise ValueError("depths cannot be empty")
        return out

    @classmethod
    def from_args(cls, args) -> "Stage0SPRConfig":
        return cls(
            trades_window_s=args.trades_window_s,
            persist_window_ms=args.persist_window_ms,
            thinning_window_s=args.thinning_window_s,
            profit_net_target_bps=args.profit_net_target_bps,
            fee_maker_bps=args.fee_maker_bps,
            fee_taker_bps=args.fee_taker_bps,
            slip_bps=args.slip_bps,
            p_spread=args.p_spread,
            p_micro=args.p_micro,
            p_obi10=args.p_obi10,
            p_ti=args.p_ti,
            p_nps=args.p_nps,
            p_thin=args.p_thin,
            p_range60s=args.p_range60s,
            persist_ms_min=args.persist_ms_min,
            ms_keep_quantile=args.ms_keep_quantile,
            depths=cls.parse_depths(getattr(args, "depths", "1,3,5,10,14")),
            max_horizon_s=args.max_horizon_s,
            tick_size=getattr(args, "tick_size", cls.tick_size),
            spread_roll_med_s=getattr(args, "spread_roll_med_s", cls.spread_roll_med_s),
            p_spread_rel=getattr(args, "p_spread_rel", cls.p_spread_rel),
            p_spread_ticks=getattr(args, "p_spread_ticks", cls.p_spread_ticks),
        )


# =============================================================================
# Utils
# =============================================================================

def _ensure_sorted(df: pd.DataFrame, col: str = "timestamp") -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if col in df.columns and not df[col].is_monotonic_increasing:
        return df.sort_values(col)
    return df


def _median_mad(x: pd.Series) -> tuple[float, float]:
    v = pd.to_numeric(x, errors="coerce").to_numpy(dtype=np.float64, copy=False)
    med = float(np.nanmedian(v))
    mad = float(np.nanmedian(np.abs(v - med)))
    return med, mad


def _robust_z(x: pd.Series, med: float, mad: float) -> pd.Series:
    return (pd.to_numeric(x, errors="coerce") - med) / (mad + 1e-12)


# =============================================================================
# Feature builders
# =============================================================================

def attach_spread_liquidity_features(
    df: pd.DataFrame,
    tick_size: float,
    roll_med_s: int = 300,
) -> pd.DataFrame:
    """
    Adds:
      - spread_ticks: (ask0-bid0)/tick_size
      - spread_ticks_1s: rolling median over last 1s (time-based)
      - spread_ticks_med_5m: rolling median over last roll_med_s seconds
      - spread_rel_5m: spread_ticks_1s / spread_ticks_med_5m
    Requires: timestamp, bid_0_price, ask_0_price
    """
    if df is None or df.empty:
        return pd.DataFrame() if df is None else df

    x = _ensure_sorted(df.copy(), "timestamp").set_index("timestamp")

    bid0 = pd.to_numeric(x["bid_0_price"], errors="coerce")
    ask0 = pd.to_numeric(x["ask_0_price"], errors="coerce")
    spr = (ask0 - bid0).replace([np.inf, -np.inf], np.nan)

    tsz = float(tick_size) if tick_size and tick_size > 0 else np.nan
    x["spread_ticks"] = (spr / (tsz + 1e-12)).astype(np.float32)

    # robust smoothing even if not exactly freq_ms
    x["spread_ticks_1s"] = x["spread_ticks"].rolling("1s", min_periods=1).median().astype(np.float32)

    win = f"{int(roll_med_s)}s"
    minp = 10
    x["spread_ticks_med_5m"] = x["spread_ticks_1s"].rolling(win, min_periods=minp).median()

    x["spread_rel_5m"] = (x["spread_ticks_1s"] / (x["spread_ticks_med_5m"] + 1e-12)).astype(np.float32)

    # safe fill at start
    x["spread_ticks_med_5m"] = x["spread_ticks_med_5m"].fillna(x["spread_ticks_1s"]).astype(np.float32)
    x["spread_rel_5m"] = (
        x["spread_rel_5m"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(1.0)
        .astype(np.float32)
    )

    return x.reset_index()


def compute_book_core_features(book: pd.DataFrame, depths: Iterable[int]) -> pd.DataFrame:
    """
    SAFE + low-memory:
    - No vstack
    - Vectorized cumulative sums for Dbid/Dask

    Requires at minimum:
      timestamp, bid_0_price, ask_0_price, bid_0_size, ask_0_size
    And for each level i used by max(depths):
      bid_i_size, ask_i_size (missing => treated as 0)
    """
    if book is None or book.empty:
        return pd.DataFrame() if book is None else book

    b = book.copy()

    bid0 = pd.to_numeric(b["bid_0_price"], errors="coerce").to_numpy(dtype=np.float64, copy=False)
    ask0 = pd.to_numeric(b["ask_0_price"], errors="coerce").to_numpy(dtype=np.float64, copy=False)

    mid = (bid0 + ask0) / 2.0
    b["mid"] = mid
    b["spread_bps"] = (ask0 - bid0) / (mid + 1e-12) * 1e4

    bid0s = pd.to_numeric(b.get("bid_0_size", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float64, copy=False)
    ask0s = pd.to_numeric(b.get("ask_0_size", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float64, copy=False)
    micro = (ask0 * bid0s + bid0 * ask0s) / (bid0s + ask0s + 1e-12)
    b["micro"] = micro
    b["micro_bias_bps"] = (micro - mid) / (mid + 1e-12) * 1e4

    depths = tuple(sorted(set(int(x) for x in depths)))
    if not depths:
        return b
    max_k = int(max(depths))

    n = len(b)
    bid_mat = np.zeros((n, max_k + 1), dtype=np.float32)
    ask_mat = np.zeros((n, max_k + 1), dtype=np.float32)

    for i in range(max_k + 1):
        colb = f"bid_{i}_size"
        cola = f"ask_{i}_size"
        if colb in b.columns:
            bid_mat[:, i] = pd.to_numeric(b[colb], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32, copy=False)
        if cola in b.columns:
            ask_mat[:, i] = pd.to_numeric(b[cola], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32, copy=False)

    bid_cum = np.cumsum(bid_mat, axis=1, dtype=np.float64)
    ask_cum = np.cumsum(ask_mat, axis=1, dtype=np.float64)

    for k in depths:
        Dbid = bid_cum[:, k]
        Dask = ask_cum[:, k]
        b[f"Dbid_{k}"] = Dbid
        b[f"Dask_{k}"] = Dask
        b[f"OBI_{k}"] = (Dbid - Dask) / (Dbid + Dask + 1e-9)

    return b


def attach_range_features(df: pd.DataFrame, freq_ms: int = 250, win_60s: int = 60, win_10m: int = 600) -> pd.DataFrame:
    """
    Adds:
      - range_60s_bps: rolling max(mid) - min(mid) over ~60 seconds (bps)
      - range_10m_bps: rolling max(mid) - min(mid) over ~10 minutes (bps)
    Assumes df has columns: timestamp, mid
    """
    if df is None or df.empty:
        return pd.DataFrame() if df is None else df

    x = _ensure_sorted(df.copy(), "timestamp")

    step_s = float(freq_ms) / 1000.0
    w60 = max(1, int(round(float(win_60s) / step_s)))
    w10m = max(1, int(round(float(win_10m) / step_s)))

    mid = pd.to_numeric(x["mid"], errors="coerce").to_numpy(dtype=np.float64, copy=False)
    s = pd.Series(mid)

    roll_max_60 = s.rolling(w60, min_periods=w60).max()
    roll_min_60 = s.rolling(w60, min_periods=w60).min()
    x["range_60s_bps"] = (roll_max_60 - roll_min_60) / (s + 1e-12) * 1e4

    if "range_10m_bps" not in x.columns:
        roll_max_10m = s.rolling(w10m, min_periods=w10m).max()
        roll_min_10m = s.rolling(w10m, min_periods=w10m).min()
        x["range_10m_bps"] = (roll_max_10m - roll_min_10m) / (s + 1e-12) * 1e4

    x["range_60s_bps"] = x["range_60s_bps"].fillna(0.0).astype(np.float32)
    x["range_10m_bps"] = x["range_10m_bps"].fillna(0.0).astype(np.float32)
    return x


def attach_persistence_features(book: pd.DataFrame, window_ms: int = 800) -> pd.DataFrame:
    if book is None or book.empty:
        return pd.DataFrame() if book is None else book

    b = _ensure_sorted(book.copy(), "timestamp")
    ts = b["timestamp"].values.astype("datetime64[ns]")

    s_micro = np.sign(pd.to_numeric(b["micro_bias_bps"], errors="coerce").fillna(0.0).values).astype(np.int8)
    s_obi10 = np.sign(pd.to_numeric(b["OBI_10"], errors="coerce").fillna(0.0).values).astype(np.int8)

    def persist_ms_from_sign(sign_arr: np.ndarray) -> np.ndarray:
        out = np.zeros(len(sign_arr), dtype=np.float32)
        seg_start = 0
        for i in range(1, len(sign_arr)):
            if sign_arr[i] == 0:
                seg_start = i
            elif sign_arr[i] != sign_arr[i - 1]:
                seg_start = i

            dt_ms = (ts[i] - ts[seg_start]) / np.timedelta64(1, "ms")
            dt_ms = float(dt_ms)
            if dt_ms < 0.0:
                dt_ms = 0.0

            out[i] = float(min(dt_ms, float(window_ms)))
        return out

    b["persist_micro_ms"] = persist_ms_from_sign(s_micro)
    b["persist_obi10_ms"] = persist_ms_from_sign(s_obi10)
    return b


def attach_thinning_features(book: pd.DataFrame, window_s: float = 2.0) -> pd.DataFrame:
    if book is None or book.empty:
        return pd.DataFrame() if book is None else book

    b = _ensure_sorted(book.copy(), "timestamp").set_index("timestamp")
    if "dir0" not in b.columns:
        raise ValueError("dir0 must exist before thinning features")

    for req in ("Dask_3", "Dbid_3", "Dask_10", "Dbid_10"):
        if req not in b.columns:
            raise ValueError(f"{req} must exist before thinning features")

    Dask_3 = b["Dask_3"]
    Dbid_3 = b["Dbid_3"]
    Dask_10 = b["Dask_10"]
    Dbid_10 = b["Dbid_10"]

    opp3 = np.where(b["dir0"].values > 0, Dask_3.values, Dbid_3.values)
    opp10 = np.where(b["dir0"].values > 0, Dask_10.values, Dbid_10.values)

    b["Dopp_3"] = opp3
    b["Dopp_10"] = opp10

    win = f"{int(float(window_s) * 1000)}ms"
    med_opp3 = b["Dopp_3"].rolling(win).median()
    med_opp10 = b["Dopp_10"].rolling(win).median()

    b["thinning_opp_3"] = (med_opp3 - b["Dopp_3"]) / (med_opp3 + 1e-9)
    b["thinning_opp_10"] = (med_opp10 - b["Dopp_10"]) / (med_opp10 + 1e-9)
    return b.reset_index()


# =============================================================================
# Thresholds + MS score
# =============================================================================

@dataclass
class SPRThresholds:
    spread_bps_max: float
    micro_abs_min: float
    obi10_abs_min: float
    ti_abs_min: float
    nps_min: float
    thin_min: float

    med_micro_abs: float
    mad_micro_abs: float
    med_obi10_abs: float
    mad_obi10_abs: float
    med_ti_abs: float
    mad_ti_abs: float
    med_thin: float
    mad_thin: float
    med_spread: float
    mad_spread: float
    med_nps: float
    mad_nps: float

    spread_ticks_max: float
    spread_rel_max: float
    range60s_min: float


def fit_thresholds(train_features: pd.DataFrame, cfg: Stage0SPRConfig) -> SPRThresholds:
    df = train_features
    if df is None or df.empty:
        raise SystemExit("[fit_thresholds] empty train_features")

    need = ["spread_ticks_1s", "spread_rel_5m", "range_60s_bps", "spread_bps", "micro_bias_bps", "OBI_10", "TI", "nps", "thinning_opp_3"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise SystemExit(f"[fit_thresholds] missing cols: {miss}")

    spread_bps_max = float(np.nanpercentile(df["spread_bps"], cfg.p_spread))

    micro_abs = pd.to_numeric(df["micro_bias_bps"], errors="coerce").abs()
    obi10_abs = pd.to_numeric(df["OBI_10"], errors="coerce").abs()
    ti_abs = pd.to_numeric(df["TI"], errors="coerce").abs()

    micro_abs_min = float(np.nanpercentile(micro_abs, cfg.p_micro))
    obi10_abs_min = float(np.nanpercentile(obi10_abs, cfg.p_obi10))
    ti_abs_min = float(np.nanpercentile(ti_abs, cfg.p_ti))
    nps_min = float(np.nanpercentile(pd.to_numeric(df["nps"], errors="coerce"), cfg.p_nps))
    thin_min = float(np.nanpercentile(pd.to_numeric(df["thinning_opp_3"], errors="coerce"), cfg.p_thin))

    med_micro_abs, mad_micro_abs = _median_mad(micro_abs)
    med_obi10_abs, mad_obi10_abs = _median_mad(obi10_abs)
    med_ti_abs, mad_ti_abs = _median_mad(ti_abs)
    med_thin, mad_thin = _median_mad(df["thinning_opp_3"])
    med_spread, mad_spread = _median_mad(df["spread_bps"])
    med_nps, mad_nps = _median_mad(df["nps"])

    spread_ticks_max = float(np.nanpercentile(df["spread_ticks_1s"], cfg.p_spread_ticks))
    spread_rel_max = float(np.nanpercentile(df["spread_rel_5m"], cfg.p_spread_rel))
    range60s_min = float(np.nanpercentile(df["range_60s_bps"], cfg.p_range60s))

    return SPRThresholds(
        spread_bps_max=spread_bps_max,
        micro_abs_min=micro_abs_min,
        obi10_abs_min=obi10_abs_min,
        ti_abs_min=ti_abs_min,
        nps_min=nps_min,
        thin_min=thin_min,
        med_micro_abs=med_micro_abs,
        mad_micro_abs=mad_micro_abs,
        med_obi10_abs=med_obi10_abs,
        mad_obi10_abs=mad_obi10_abs,
        med_ti_abs=med_ti_abs,
        mad_ti_abs=mad_ti_abs,
        med_thin=med_thin,
        mad_thin=mad_thin,
        med_spread=med_spread,
        mad_spread=mad_spread,
        med_nps=med_nps,
        mad_nps=mad_nps,
        spread_ticks_max=spread_ticks_max,
        spread_rel_max=spread_rel_max,
        range60s_min=range60s_min,
    )


def compute_ms(df: pd.DataFrame, thr: SPRThresholds) -> pd.Series:
    micro_abs = pd.to_numeric(df["micro_bias_bps"], errors="coerce").abs()
    obi10_abs = pd.to_numeric(df["OBI_10"], errors="coerce").abs()
    ti_abs = pd.to_numeric(df["TI"], errors="coerce").abs()

    z_micro = _robust_z(micro_abs, thr.med_micro_abs, thr.mad_micro_abs)
    z_obi10 = _robust_z(obi10_abs, thr.med_obi10_abs, thr.mad_obi10_abs)
    z_ti = _robust_z(ti_abs, thr.med_ti_abs, thr.mad_ti_abs)
    z_thin = _robust_z(df["thinning_opp_3"], thr.med_thin, thr.mad_thin)
    z_spread = _robust_z(df["spread_bps"], thr.med_spread, thr.mad_spread)
    z_nps = _robust_z(df["nps"], thr.med_nps, thr.mad_nps)

    return (1.0 * z_micro + 0.9 * z_obi10 + 0.9 * z_ti + 0.7 * z_thin - 0.4 * z_spread + 0.2 * z_nps)


# =============================================================================
# Labeling (t2 endogenous)
# =============================================================================

def label_resolution_endogenous(
    candidates: pd.DataFrame,
    book_features: pd.DataFrame,
    cfg: "Stage0SPRConfig",
    p_spread_exp: float = 99.0,
    cont_win: int = 3,
) -> pd.DataFrame:
    """
    HIT (label=1): first time pnl_net >= cfg.profit_net_target_bps within horizon.
    RESOLUTION onset (t_res): earliest of (spread expansion+continuation) or hit itself.
    Pricing:
      - long entry at ask0, exit at bid0
      - short entry at bid0, exit at ask0
    """
    if candidates is None or len(candidates) == 0:
        out = candidates.copy() if candidates is not None else pd.DataFrame()
        out["label"] = np.int8(0)
        out["reason_res"] = ""
        out["reason_hit"] = ""
        out["t_res"] = np.datetime64("NaT")
        out["t_hit"] = np.datetime64("NaT")
        out["time_to_res_s"] = np.nan
        out["time_to_hit_s"] = np.nan
        out["pnl_raw_at_res_bps"] = np.nan
        out["pnl_net_at_res_bps"] = np.nan
        out["pnl_net_at_hit_bps"] = np.nan
        out["pnl_raw_max_bps"] = np.nan
        out["pnl_net_max_bps"] = np.nan
        out["pnl_raw_min_bps"] = np.nan
        out["pnl_net_min_bps"] = np.nan
        out["target_net_bps"] = float(getattr(cfg, "profit_net_target_bps", np.nan))
        out["fees_rt_bps"] = float(getattr(cfg, "fees_rt_bps", np.nan))
        out["slip_bps"] = float(getattr(cfg, "slip_bps", np.nan))
        out["spread_exp_thr_bps"] = np.nan
        return out

    need_b = {"timestamp", "bid_0_price", "ask_0_price", "spread_bps"}
    missing_b = [c for c in need_b if c not in book_features.columns]
    if missing_b:
        raise SystemExit(f"[label] book_features missing columns: {missing_b}")

    if "timestamp" not in candidates.columns or "dir0" not in candidates.columns:
        raise SystemExit("[label] candidates must have columns: ['timestamp','dir0']")

    b = book_features[["timestamp", "bid_0_price", "ask_0_price", "spread_bps"]].copy()
    b = _ensure_sorted(b, "timestamp").reset_index(drop=True)

    tmax = np.timedelta64(int(float(cfg.max_horizon_s) * 1e9), "ns")

    target_net = float(cfg.profit_net_target_bps)
    fees_rt = float(cfg.fees_rt_bps)
    slip = float(cfg.slip_bps)
    cost_bps = fees_rt + slip

    ts_arr = b["timestamp"].values.astype("datetime64[ns]")
    bid0 = pd.to_numeric(b["bid_0_price"], errors="coerce").to_numpy(dtype=np.float64, copy=False)
    ask0 = pd.to_numeric(b["ask_0_price"], errors="coerce").to_numpy(dtype=np.float64, copy=False)
    spread = pd.to_numeric(b["spread_bps"], errors="coerce").to_numpy(dtype=np.float64, copy=False)

    q_thr = float(np.nanpercentile(spread, p_spread_exp))
    med = float(np.nanmedian(spread))
    mad = float(np.nanmedian(np.abs(spread - med)))
    robust_thr = med + 5.0 * mad
    spread_thr = max(q_thr, robust_thr)

    cand = _ensure_sorted(candidates.copy(), "timestamp").reset_index(drop=True)
    cand_ts = cand["timestamp"].values.astype("datetime64[ns]")
    idx0 = np.searchsorted(ts_arr, cand_ts, side="left")

    n = len(cand)
    label = np.zeros(n, dtype=np.int8)

    reason_res = np.array([""] * n, dtype=object)
    reason_hit = np.array([""] * n, dtype=object)

    t_res = np.array([np.datetime64("NaT")] * n, dtype="datetime64[ns]")
    t_hit = np.array([np.datetime64("NaT")] * n, dtype="datetime64[ns]")

    time_to_res_s = np.full(n, np.nan, dtype=np.float64)
    time_to_hit_s = np.full(n, np.nan, dtype=np.float64)

    pnl_raw_at_res = np.full(n, np.nan, dtype=np.float64)
    pnl_net_at_res = np.full(n, np.nan, dtype=np.float64)
    pnl_net_at_hit = np.full(n, np.nan, dtype=np.float64)

    pnl_raw_max = np.full(n, np.nan, dtype=np.float64)
    pnl_net_max = np.full(n, np.nan, dtype=np.float64)
    pnl_raw_min = np.full(n, np.nan, dtype=np.float64)
    pnl_net_min = np.full(n, np.nan, dtype=np.float64)

    for j in range(n):
        i0 = int(idx0[j])
        if i0 >= len(ts_arr):
            continue

        t0 = ts_arr[i0]
        dir0 = int(cand.loc[j, "dir0"])
        if dir0 == 0:
            continue

        entry = ask0[i0] if dir0 > 0 else bid0[i0]
        if not np.isfinite(entry) or entry <= 0:
            continue

        t_end = t0 + tmax
        i1 = int(np.searchsorted(ts_arr, t_end, side="right"))
        if i1 <= i0 + 1:
            continue

        bid_path = bid0[i0:i1]
        ask_path = ask0[i0:i1]

        if dir0 > 0:
            pnl_raw_path = (bid_path - entry) / entry * 1e4
        else:
            pnl_raw_path = (entry - ask_path) / entry * 1e4

        pnl_net_path = pnl_raw_path - cost_bps

        if pnl_raw_path.size:
            pnl_raw_max[j] = float(np.nanmax(pnl_raw_path))
            pnl_net_max[j] = float(np.nanmax(pnl_net_path))
            pnl_raw_min[j] = float(np.nanmin(pnl_raw_path))
            pnl_net_min[j] = float(np.nanmin(pnl_net_path))

        hit_idx = np.where(pnl_net_path >= target_net)[0]
        k_hit = int(hit_idx[0]) if hit_idx.size else None

        sp = spread[i0:i1]
        exp_idx = np.where(sp >= spread_thr)[0]
        k_sp = None
        if exp_idx.size:
            k0 = int(exp_idx[0])
            k1c = min(k0 + int(cont_win), len(pnl_net_path))
            if k1c > k0 + 1:
                if (
                    np.isfinite(pnl_net_path[k1c - 1])
                    and np.isfinite(pnl_net_path[k0])
                    and (pnl_net_path[k1c - 1] > pnl_net_path[k0])
                ):
                    k_sp = k0

        k_res = None
        r_res = None
        if k_hit is not None:
            k_res = k_hit
            r_res = "pnl_net"
        if k_sp is not None and (k_res is None or k_sp < k_res):
            k_res = k_sp
            r_res = "spread_exp"

        if k_res is not None:
            t_r = ts_arr[i0 + k_res]
            t_res[j] = t_r
            time_to_res_s[j] = float((t_r - t0) / np.timedelta64(1, "s"))
            pnl_raw_at_res[j] = float(pnl_raw_path[k_res])
            pnl_net_at_res[j] = float(pnl_net_path[k_res])
            reason_res[j] = r_res

        if k_hit is not None:
            label[j] = 1
            t_h = ts_arr[i0 + k_hit]
            t_hit[j] = t_h
            time_to_hit_s[j] = float((t_h - t0) / np.timedelta64(1, "s"))
            pnl_net_at_hit[j] = float(pnl_net_path[k_hit])
            reason_hit[j] = "pnl_net"

    out = cand.copy()
    out["label"] = label
    out["reason_res"] = reason_res
    out["t_res"] = t_res
    out["time_to_res_s"] = time_to_res_s
    out["pnl_raw_at_res_bps"] = pnl_raw_at_res
    out["pnl_net_at_res_bps"] = pnl_net_at_res

    out["reason_hit"] = reason_hit
    out["t_hit"] = t_hit
    out["time_to_hit_s"] = time_to_hit_s
    out["pnl_net_at_hit_bps"] = pnl_net_at_hit

    out["pnl_raw_max_bps"] = pnl_raw_max
    out["pnl_net_max_bps"] = pnl_net_max
    out["pnl_raw_min_bps"] = pnl_raw_min
    out["pnl_net_min_bps"] = pnl_net_min

    out["target_net_bps"] = target_net
    out["fees_rt_bps"] = fees_rt
    out["slip_bps"] = slip
    out["spread_exp_thr_bps"] = spread_thr
    return out


# =============================================================================
# Candidate filters + audit
# =============================================================================

def tag_candidates(
    features: pd.DataFrame,
    cfg: Stage0SPRConfig,
    thr: SPRThresholds,
    *,
    compute_ms_score: bool = True,
) -> pd.DataFrame:
    """
    Ne DROP rien.
    Ajoute des colonnes pass_* + MS + ms_cut_value + pass_ms_keep
    """
    if features is None or len(features) == 0:
        return pd.DataFrame()

    df = features.copy(deep=False)

    # Ensure dir0 exists (direction)
    if "dir0" not in df.columns:
        df["dir0"] = np.sign(df["TI"].values + 0.5 * np.sign(df["micro_bias_bps"].values)).astype(np.int8)
    else:
        df["dir0"] = pd.to_numeric(df["dir0"], errors="coerce").fillna(0).astype(np.int8)

    # 1) Hard masks
    masks = candidate_filter_masks(df, cfg, thr)  # OrderedDict[str, np.ndarray] (order = cascade order)
    n = len(df)
    df["n_total"] = np.int32(n)

    # mapping: original mask keys -> suffix used in columns
    name_map = {
        "dir0!=0": "dir0",
        "spr_rel": "spr_rel",
        "spr_ticks": "spr_ticks",
        "range60s": "range60",
        "obi10": "obi10",
        "micro": "micro",
        "Ntot>0": "Ntot",
        "TI": "TI",
        "nps": "nps",
        "persist": "persist",
        "thin": "thin",
        "sign_micro": "sign_micro",
        "sign_ti": "sign_ti",
    }

    # per-step + cumulative funnel
    cum = np.ones(n, dtype=bool)

    for k, cond in masks.items():
        short = name_map.get(k, k)
        step = cond.astype(bool, copy=False)

        # individual pass
        df[f"pass_{short}"] = step
        df[f"n_pass_{short}"] = np.int32(step.sum())

        # cumulative pass up to this step (cascade)
        cum &= step
        df[f"pass_up_to_{short}"] = cum.copy()
        df[f"n_up_to_{short}"] = np.int32(cum.sum())

    df["pass_all_hard"] = cum
    df["n_pass_all_hard"] = np.int32(cum.sum())
    

    # 2) MS score + keep quantile (but do not drop)
    df["MS"] = np.nan
    df["ms_cut_value"] = np.nan
    df["pass_ms_keep"] = False

    if compute_ms_score:
        # MS only makes sense on rows that passed hard filters (otherwise z-scores can be noisy)
        ok = df["pass_all_hard"].values
        if ok.any():
            ms = compute_ms(df.loc[ok], thr).astype("float64")
            df.loc[ok, "MS"] = ms.values

            # compute cut on rows where MS is defined (hard-pass)
            cut = float(np.nanpercentile(ms.values, cfg.ms_keep_quantile))
            df["ms_cut_value"] = cut
            df.loc[ok, "pass_ms_keep"] = (df.loc[ok, "MS"].values >= cut)
        else:
            # no row passes hard filters -> nothing kept
            df["ms_cut_value"] = np.nan
            df["pass_ms_keep"] = False

    # Optional: attach config/thr snapshots as scalar columns (super handy in parquet)
    df["cfg_ms_keep_quantile"] = float(cfg.ms_keep_quantile)
    df["thr_range60s_min"] = float(thr.range60s_min)
    df["thr_micro_abs_min"] = float(thr.micro_abs_min)
    df["thr_obi10_abs_min"] = float(thr.obi10_abs_min)

    # keep attrs too
    df.attrs["cfg"] = asdict(cfg)
    df.attrs["thr"] = asdict(thr)

    return df

def candidate_filter_masks(df: pd.DataFrame, cfg: Stage0SPRConfig, thr: SPRThresholds) -> "OrderedDict[str, np.ndarray]":
    need = [
        "dir0",
        "TI",
        "micro_bias_bps",
        "spread_rel_5m",
        "spread_ticks_1s",
        "range_60s_bps",
        "OBI_10",
        "Ntot",
        "nps",
        "persist_micro_ms",
        "persist_obi10_ms",
        "thinning_opp_3",
    ]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise SystemExit(f"[cands] missing cols: {miss}")

    dir0 = pd.to_numeric(df["dir0"], errors="coerce").fillna(0).to_numpy(dtype=np.int8, copy=False)

    micro_sign = np.sign(pd.to_numeric(df["micro_bias_bps"], errors="coerce").fillna(0.0).values).astype(np.int8)
    ti_sign = np.sign(pd.to_numeric(df["TI"], errors="coerce").fillna(0.0).values).astype(np.int8)

    masks = OrderedDict()
    masks["dir0!=0"] = (dir0 != 0)

    masks["spr_rel"] = (pd.to_numeric(df["spread_rel_5m"], errors="coerce").values <= float(thr.spread_rel_max))
    masks["spr_ticks"] = (pd.to_numeric(df["spread_ticks_1s"], errors="coerce").values <= float(thr.spread_ticks_max))

    if "thr_range60s_min_train" in df.columns:
        thr_series = pd.to_numeric(df["thr_range60s_min_train"], errors="coerce")
        thr_series = thr_series.fillna(float(thr.range60s_min))
    else:
        thr_series = pd.Series(float(thr.range60s_min), index=df.index)

    masks["range60s"] = (
        pd.to_numeric(df["range_60s_bps"], errors="coerce").values >= thr_series.values
    )

    masks["obi10"] = (np.abs(pd.to_numeric(df["OBI_10"], errors="coerce").values) >= float(thr.obi10_abs_min))
    masks["micro"] = (np.abs(pd.to_numeric(df["micro_bias_bps"], errors="coerce").values) >= float(thr.micro_abs_min))

    masks["Ntot>0"] = (pd.to_numeric(df["Ntot"], errors="coerce").fillna(0.0).values > 0.0)
    masks["TI"] = (np.abs(pd.to_numeric(df["TI"], errors="coerce").values) >= float(thr.ti_abs_min))
    masks["nps"] = (pd.to_numeric(df["nps"], errors="coerce").values >= float(thr.nps_min))

    masks["persist"] = (
        (pd.to_numeric(df["persist_micro_ms"], errors="coerce").fillna(0.0).values >= float(cfg.persist_ms_min))
        | (pd.to_numeric(df["persist_obi10_ms"], errors="coerce").fillna(0.0).values >= float(cfg.persist_ms_min))
    )
    masks["thin"] = (pd.to_numeric(df["thinning_opp_3"], errors="coerce").values >= float(thr.thin_min))

    masks["sign_micro"] = (micro_sign == dir0)
    masks["sign_ti"] = (ti_sign == dir0)

    return masks

def apply_candidate_filters(df: pd.DataFrame, masks: "OrderedDict[str, np.ndarray]") -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame() if df is None else df

    m = np.ones(len(df), dtype=bool)
    for _, cond in masks.items():
        m &= cond
        if not m.any():
            return df.iloc[0:0].copy()
    return df.loc[m].copy()
