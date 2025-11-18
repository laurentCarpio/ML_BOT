# next_bot/strategy/micro_decider.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import numpy as np
import pandas as pd

@dataclass(frozen=True)

class MicroParams:
    swept_levels_sec: int = 5            # fixé
    swept_min_levels: int = 1            # = 1 n'augmente pas le nombre de trade sur 2023 et 2024

    # --- resserrés pour maker-in (POST_ONLY) ---
    max_spread_bps: float = 1.20        # old 2.0
    spread_widen_redflag: float = 1.50    # fixé  

    obi_pre_thr: float = 0.48           # old 0.25

    # agressivité “now” : fenêtre sûre pour ne pas rater le fill ni se faire cueillir
    aggr_now_min: float = 0.38          # old 0.40  (barrer les rafales adverses trop marquées)
    aggr_now_max: float = 0.70          # old 0.60

    max_reversal_aggr: float = 0.60     # old 0.55

    # “mur” au meilleur
    min_wall_share_same: float = 0.55   # old 0.35 
    max_wall_share_opp: float = 0.35    # old 0.45
    require_wall_persist: bool = True

    # proxy liquidité de sortie (taker-out)
    exit_best_qty_mult: float = 1.0      # fixé (≥ médiane 60s côté sortie)

    # seuils alternatifs si le book est ultra dominant
    alt_wall_same_for_no_swept: float = 0.85   # ≥85% du top côté signal
    alt_obi_for_no_swept: float       = 0.80   # OBI très haut
    alt_aggr_min_for_no_swept: float  = 0.45   # un minimum d'agressifs dans le bon sens

# ---------------- helpers ----------------

# --- helper: évolution du share du mur opposé sur les 2 derniers snapshots pré-t0

def _count_up_steps(series: pd.Series) -> int:
    s = series.dropna().astype(float)
    if s.empty: return 0
    # compresse les runs constants → ne garde que les changements réels
    s = s[s.diff().fillna(0) != 0]
    return int((s.diff() > 0).sum())

def _count_down_steps(series: pd.Series) -> int:
    s = series.dropna().astype(float)
    if s.empty: return 0
    s = s[s.diff().fillna(0) != 0]
    return int((s.diff() < 0).sum())

def _norm_side(side: str) -> str:
    s = (side or "").strip().lower()
    if s not in ("buy", "sell"):
        raise ValueError(f"side invalide: {side!r} (attendu: 'buy' ou 'sell')")
    return s

def _swept_levels(l1: pd.DataFrame, side: str, t0, horizon_s: int) -> int:
    if l1 is None or l1.empty:
        return 0
    post = l1[(l1["timestamp"] >= t0) & (l1["timestamp"] <= t0 + pd.Timedelta(seconds=horizon_s))]
    if post.empty:
        return 0
    if side == "buy":
        # compter les marches ASK réellement franchies (sans ping-pong)
        return _count_up_steps(post["best_ask"])
    else:
        # compter les marches BID réellement franchies (sans ping-pong)
        return _count_down_steps(post["best_bid"])

def _aggr_ratio(tr: pd.DataFrame, side: str, t_start, t_end) -> float:
    if tr is None or tr.empty:
        return np.nan
    x = tr[(tr["timestamp"] >= t_start) & (tr["timestamp"] <= t_end)]
    if x.empty:
        return np.nan
    if side == "buy":
        num = x["is_aggr_buy"].astype(bool).sum()
    else:
        num = (~x["is_aggr_buy"].astype(bool)).sum()
    den = len(x)
    return float(num) / float(den) if den > 0 else np.nan

def _net_delta(tr: pd.DataFrame, t_start, t_end) -> float:
    if tr is None or tr.empty:
        return 0.0
    x = tr[(tr["timestamp"] >= t_start) & (tr["timestamp"] <= t_end)]
    if x.empty:
        return 0.0
    buy_qty  = x.loc[x["is_aggr_buy"].astype(bool), "qty"].astype(float).sum()
    sell_qty = x.loc[~x["is_aggr_buy"].astype(bool), "qty"].astype(float).sum()
    return float(buy_qty - sell_qty)

# ---------------- features & decision ----------------

def compute_features(tr_win: pd.DataFrame, l1_win: pd.DataFrame, *, side: str, t0, entry: float = np.nan, params: MicroParams = None) -> Dict:
    """
    Calcule les features consommées par decide_micro(...). Fenêtre courte autour de t0.
    """
    P = params or MicroParams()
    side = _norm_side(side)

    feats: Dict[str, object] = {}
    feats["side"] = side 

    # t0 → UTC tz-aware
    t0 = pd.Timestamp(t0, tz="UTC") if pd.Timestamp(t0).tzinfo is None else pd.Timestamp(t0).tz_convert("UTC")

    # sélections
    l1_pre  = l1_win[l1_win["timestamp"] <= t0]
    l1_last = l1_pre.iloc[[-1]] if not l1_pre.empty else l1_win.tail(1)

    feats["_l1_pre_view"] = l1_pre  # petite vue (fenêtre courte), utile au delta de mur

    if l1_last.empty:
        feats.update({
            "spread_bps": np.nan,
            "obi_pre": np.nan,
            "spread_widen": np.nan,
            "wall_same_share": 0.0,
            "wall_opp_share": 0.0,
            "wall_persist": False,
            "best_qty_vs_median60": np.nan,
            "aggr_ratio_now": np.nan,
            "net_delta_now": 0.0,
            "swept_lvls": 0,
            "reversal_aggr": np.nan,
            "S_breakout": 0, "S_trap": 0,
            "iceberg_opp": False,
        })
        return feats

    # 1) Spread (bps) = (ask - bid) / mid * 1e4
    bb = float(l1_last["best_bid"].values[0])
    ba = float(l1_last["best_ask"].values[0])
    mid = 0.5 * (bb + ba) if np.isfinite(bb) and np.isfinite(ba) else np.nan
    spread = (ba - bb) if np.isfinite(bb) and np.isfinite(ba) else np.nan
    feats["spread_bps"] = (spread / mid * 1e4) if (np.isfinite(spread) and mid > 0) else np.nan

    # 2) OBI “pre” (top1, instantané) dans [-1,1]
    bid_qty = float(l1_last["bid_qty"].values[0])
    ask_qty = float(l1_last["ask_qty"].values[0])
    den = (bid_qty + ask_qty)
    obi_pre = ((bid_qty - ask_qty) / den) if den > 0 else np.nan
    feats["obi_pre"] = float(np.clip(obi_pre, -1.0, 1.0)) if np.isfinite(obi_pre) else np.nan

    # 3) Spread widening (vs médiane pré-t0)
    if not l1_pre.empty:
        pre_spreads = (l1_pre["best_ask"] - l1_pre["best_bid"]).astype(float)
        med_spread  = float(pre_spreads.median()) if len(pre_spreads) else np.nan
        feats["spread_widen"] = (spread / med_spread) if (np.isfinite(spread) and med_spread > 0) else np.nan
    else:
        feats["spread_widen"] = np.nan

    # 4) Walls
    total_top_qty = bid_qty + ask_qty
    if total_top_qty > 0:
        if side == "buy":
            feats["wall_same_share"] = bid_qty / total_top_qty
            feats["wall_opp_share"]  = ask_qty / total_top_qty
        else:
            feats["wall_same_share"] = ask_qty / total_top_qty
            feats["wall_opp_share"]  = bid_qty / total_top_qty
    else:
        feats["wall_same_share"] = 0.0
        feats["wall_opp_share"]  = 0.0

        # --- persistance du mur (vue pré-t0 plus robuste) ---
    wall_persist = False
    if not l1_pre.empty:
        # on regarde les ~4 derniers snapshots (quelques centaines de ms typiquement)
        window = l1_pre.tail(4).copy()
        den = (window["bid_qty"] + window["ask_qty"]).replace(0, np.nan)
        if side == "buy":
            share = (window["bid_qty"] / den).astype(float).fillna(0.0)
        else:
            share = (window["ask_qty"] / den).astype(float).fillna(0.0)

        last_ok = bool(len(share) > 0 and share.iloc[-1] >= float(MicroParams.min_wall_share_same))
        mono_ok = bool(getattr(share, "is_monotonic_increasing", False))
        # Accepte une micro-baisse unique (anti faux négatif) :
        if not mono_ok and len(share) >= 3:
            diffs = share.diff().fillna(0.0)
            mono_ok = bool((diffs < 0).sum() <= 1 and share.iloc[-1] >= share.iloc[0])

        wall_persist = bool(last_ok and mono_ok)

    feats["wall_persist"] = wall_persist

    # 5) Proxy liquidité de sortie (taker-out)
    if not l1_pre.empty:
        if side == "buy":
            series = l1_pre["bid_qty"]; nowq = bid_qty   # sortie au bid
        else:
            series = l1_pre["ask_qty"]; nowq = ask_qty   # sortie à l'ask
        med = float(series.median()) if len(series) else np.nan
        feats["best_qty_vs_median60"] = (nowq / med) if (np.isfinite(nowq) and med > 0) else np.nan
    else:
        feats["best_qty_vs_median60"] = np.nan

    # 6) Aggressivité “now” + net delta (petit lookback)
    if tr_win is not None and not tr_win.empty:
        lookback = pd.Timedelta(seconds=min(5, max(1, int(P.swept_levels_sec))))  # 1–5s
        t_start = max(tr_win["timestamp"].min(), t0 - lookback)
        feats["aggr_ratio_now"] = _aggr_ratio(tr_win, side, t_start, t0)
        feats["net_delta_now"]  = _net_delta(tr_win, t_start, t0)
    else:
        feats["aggr_ratio_now"] = np.nan
        feats["net_delta_now"]  = 0.0

    # 7) Swept levels (post-t0 court)
    feats["swept_lvls"] = _swept_levels(l1_win, side, t0, P.swept_levels_sec)

    # 8) Reversal aggressiveness (opposé dans [t0, t0+3s])
    if tr_win is not None and not tr_win.empty:
        t1 = t0 + pd.Timedelta(seconds=3)
        aggr_in = _aggr_ratio(tr_win, side, t0, t1)        # pro-directionnel
        feats["reversal_aggr"] = (1.0 - aggr_in) if np.isfinite(aggr_in) else np.nan
    else:
        feats["reversal_aggr"] = np.nan

    # 9) Placeholders/compat
    feats.setdefault("S_breakout", 0)
    feats.setdefault("S_trap", 0)
    feats.setdefault("iceberg_opp", False)

    return feats

def decide_micro(feats, P: MicroParams, trap_mode: str = "off"):
    """
    Retourne: (decision, reason, S_breakout, S_trap, mode)
    mode ∈ {"BREAK","NO_MODE"}
    (TRAP retiré – param trap_mode ignoré pour compat)
    """
    # --------- valeurs calculées par compute_features(...) ---------
    spread_bps      = feats["spread_bps"]
    obi_pre         = feats["obi_pre"]
    spread_widen    = feats["spread_widen"]
    aggr_now        = feats["aggr_ratio_now"]
    reversal_aggr   = feats["reversal_aggr"]
    wall_same_share = feats.get("wall_same_share", 0.0)
    wall_opp_share  = feats.get("wall_opp_share",  0.0)
    wall_persist    = feats.get("wall_persist", False)
    best_qty_vs_med = feats.get("best_qty_vs_median60", 1.0)
    swept_lvls      = feats.get("swept_lvls", 0)
    S_breakout      = feats.get("S_breakout", 0)
    S_trap          = feats.get("S_trap", 0)   # laissé tel quel pour compat de signature

    # ======================= Relâchements contextuels (locaux) =======================
    # On ne modifie PAS l’objet P (dataclass frozen) → variables locales
    local_aggr_now_min   = P.aggr_now_min
    local_max_spread_bps = P.max_spread_bps

    # 1) Book ultra dominant → tolère aggr_now_min = 0.40
    if wall_same_share >= 0.90:
        local_aggr_now_min = min(local_aggr_now_min, 0.40)

    # 2) OBI très haut → tolère un spread un peu plus large (jusqu’à 1.20 bps)
    if np.isfinite(obi_pre) and obi_pre >= 0.90:
        local_max_spread_bps = max(local_max_spread_bps, 1.20)

    # ======================= Clauses swept & fallback domination book =======================
    swept_ok = (swept_lvls >= P.swept_min_levels)

    dom_book_ok = (
        np.isfinite(obi_pre) and
        (wall_same_share >= P.alt_wall_same_for_no_swept) and
        (obi_pre        >= P.alt_obi_for_no_swept) and
        np.isfinite(aggr_now) and (aggr_now >= P.alt_aggr_min_for_no_swept)
    )

    # ======================= BREAKOUT (maker-in) =======================
    if np.isfinite(spread_bps) and spread_bps <= local_max_spread_bps \
       and np.isfinite(obi_pre) and obi_pre >= P.obi_pre_thr \
       and (not np.isfinite(spread_widen) or spread_widen <= P.spread_widen_redflag) \
       and np.isfinite(aggr_now) and (local_aggr_now_min <= aggr_now <= P.aggr_now_max) \
       and wall_same_share >= P.min_wall_share_same \
       and wall_opp_share  <= P.max_wall_share_opp \
       and (not P.require_wall_persist or wall_persist) \
       and (not np.isfinite(reversal_aggr) or reversal_aggr <= P.max_reversal_aggr) \
       and best_qty_vs_med >= P.exit_best_qty_mult \
       and (swept_ok or dom_book_ok):
        # raison explicite utile en debug
        reason = "maker_safe" if swept_ok else "maker_safe_no_swept_but_dominant_book"
        return "GO", reason, S_breakout, S_trap, "BREAK"

    # ======================= Rien de valide =======================
    return "NO_GO", "rules_not_met", S_breakout, S_trap, "NO_MODE"