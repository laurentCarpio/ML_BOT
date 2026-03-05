from __future__ import annotations
import numpy as np
import pandas as pd

def pnl_at_T_bps(
    events: pd.DataFrame,
    book: pd.DataFrame,
    dir_series: pd.Series,
    T: int,
    tol_s: float,
    return_audit: bool = False,
):
    """
    Returns:
      - pnl_net_bps (pd.Series)
      - optionally audit dict if return_audit=True
    """
    e = events.copy()
    e["_dir"] = dir_series.astype("int64").values
    e["t_entry"] = pd.to_datetime(e["timestamp"], utc=True)
    e["t_exit"]  = e["t_entry"] + pd.to_timedelta(int(T), unit="s")

    b = book[["timestamp", "mid"]].dropna().sort_values("timestamp").copy()
    b["timestamp"] = pd.to_datetime(b["timestamp"], utc=True)

    tol = pd.Timedelta(seconds=float(tol_s))

    # --- entry ---
    entry = pd.merge_asof(
        e[["t_entry"]].rename(columns={"t_entry": "timestamp"}).sort_values("timestamp"),
        b, on="timestamp", direction="nearest", tolerance=tol
    ).rename(columns={"mid": "p_entry"})
    # --- exit ---
    exit_ = pd.merge_asof(
        e[["t_exit"]].rename(columns={"t_exit": "timestamp"}).sort_values("timestamp"),
        b, on="timestamp", direction="nearest", tolerance=tol
    ).rename(columns={"mid": "p_exit"})

    # restore original event order
    entry = entry.reindex(e.index)
    exit_  = exit_.reindex(e.index)

    p_entry = entry["p_entry"]
    p_exit  = exit_["p_exit"]
    d       = e["_dir"]

    bad = p_entry.isna() | p_exit.isna() | (p_entry <= 0) | (p_exit <= 0) | (d == 0)

    pnl_raw_bps = (d * (p_exit / p_entry - 1.0)) * 1e4
    pnl_net_bps = pnl_raw_bps - e["fees_rt_bps"].astype("float64") - e["slip_bps"].astype("float64")
    pnl_net_bps = pnl_net_bps.astype("float64")
    pnl_net_bps[bad] = np.nan

    if not return_audit:
        return pnl_net_bps

    # ---- AUDIT dt(ms) ----
    # We have the merge_asof output timestamps in entry/exit_["timestamp"] (matched book ts)
    # and requested timestamps are e["t_entry"]/e["t_exit"].
    # When merge_asof fails, matched timestamp is NaT.
    te_req = e["t_entry"]
    tx_req = e["t_exit"]
    te_hit = pd.to_datetime(entry["timestamp"], utc=True, errors="coerce")
    tx_hit = pd.to_datetime(exit_["timestamp"],  utc=True, errors="coerce")

    def _dt_ms(req, hit):
        dt = (hit - req)
        # dt is timedelta; convert to ms float
        return (dt / np.timedelta64(1, "ms")).astype("float64")

    dt_entry_ms = _dt_ms(te_req, te_hit)
    dt_exit_ms  = _dt_ms(tx_req, tx_hit)

    audit = {
        "T": float(T),
        "n_events": float(len(e)),
        "tol_s": float(tol_s),
        "na_entry_rate": float(pd.isna(p_entry).mean()),
        "na_exit_rate":  float(pd.isna(p_exit).mean()),
        "ok_rate":       float((~bad).mean()),
        "entry_dt_ms_abs_p50": float(np.nanmedian(np.abs(dt_entry_ms))) if np.isfinite(np.nanmedian(np.abs(dt_entry_ms))) else np.nan,
        "entry_dt_ms_abs_p95": float(np.nanquantile(np.abs(dt_entry_ms), 0.95)) if np.isfinite(np.nanquantile(np.abs(dt_entry_ms), 0.95)) else np.nan,
        "entry_dt_ms_abs_max": float(np.nanmax(np.abs(dt_entry_ms))) if np.isfinite(np.nanmax(np.abs(dt_entry_ms))) else np.nan,
        "exit_dt_ms_abs_p50":  float(np.nanmedian(np.abs(dt_exit_ms))) if np.isfinite(np.nanmedian(np.abs(dt_exit_ms))) else np.nan,
        "exit_dt_ms_abs_p95":  float(np.nanquantile(np.abs(dt_exit_ms), 0.95)) if np.isfinite(np.nanquantile(np.abs(dt_exit_ms), 0.95)) else np.nan,
        "exit_dt_ms_abs_max":  float(np.nanmax(np.abs(dt_exit_ms))) if np.isfinite(np.nanmax(np.abs(dt_exit_ms))) else np.nan,
        "book_ts_min": float(pd.to_datetime(b["timestamp"].min(), utc=True).timestamp()) if len(b) else np.nan,
        "book_ts_max": float(pd.to_datetime(b["timestamp"].max(), utc=True).timestamp()) if len(b) else np.nan,
    }

    return pnl_net_bps, audit

def apply_exit_rule(pnl_45, pnl_60, pnl_120, pnl_180, rule: str):
    if rule == "exit:60_if_neg_else_180":
        return np.where(pnl_60 < 0, pnl_60, pnl_180)
    if rule == "exit:60_if_neg_else_120":
        return np.where(pnl_60 < 0, pnl_60, pnl_120)
    if rule == "exit:45_if_neg_else_180":
        return np.where(pnl_45 < 0, pnl_45, pnl_180)
    raise ValueError(f"Unknown rule: {rule}")