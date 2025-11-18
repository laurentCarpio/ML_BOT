# next_bot/strategy/v1_breakout.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, Iterable
import numpy as np
import pandas as pd
import math

# On dépend explicitement des indicateurs/constantes produits par prepare_data.
# AUCUN fallback: si ces colonnes ne sont pas présentes, on lève une erreur.
from .prepare_data import (
    SLOPE_MIN,         # ~0.05%/barre
    EXT_MAX_ATR,       # extension max vs SMA20 au signal (en ATR)
    TOUCH_BAND,        # tolérance de "retour vers 20" (en ATR/close)
    PIVOT_LEN,         # minibase (ex: 3)
)

# ---------------------------------------------------------------------
# Petites helpers
# ---------------------------------------------------------------------
def _require_cols(df: pd.DataFrame, cols: Iterable[str], ctx: str):
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise ValueError(
            f"[{ctx}] colonnes manquantes {miss}. "
            f"Tu dois appeler prepare_data.calculate_indicators(...) avant, pour enrichir le DataFrame."
        )

def _last(df: Optional[pd.DataFrame]) -> Optional[pd.Series]:
    return None if df is None or df.empty else df.iloc[-1]

# ---------------------------------------------------------------------
# Stratégie V1 Breakout (ABCD retail) — sans fallback
# ---------------------------------------------------------------------
@dataclass
class V1BreakoutStrategy:
    symbol: str
    frequency: str
    _ctx: Dict[str, Any] = None

    # ⚠️ NO FALLBACKS — on préfère planter si une méthode est manquante
    def __init__(self, symbol: str, frequency: str, *, slope_min: float = 2e-4, soft_A: bool = True):
        self.symbol = symbol
        self.frequency = frequency
        self._ctx = {}
        self.last_gate_reason: Optional[str] = None
        # Nouveaux paramètres
        self.slope_min = float(slope_min)
        self.soft_A = bool(soft_A)

    # Optionnel: contexte HTF injecté depuis run_replay (ex: df 1h indexé par timestamp)
    def set_context(self, ctx: Dict[str, Any]) -> None:
        self._ctx = (ctx or {})

    def precompute_full(self, df: pd.DataFrame, f: str) -> pd.DataFrame:
        """Hook optionnel (no-op). Les colonnes doivent venir de prepare_data."""
        return df

    def enrich_with_mtf(self, df: pd.DataFrame) -> pd.DataFrame:
        """Hook optionnel (no-op)."""
        return df

    # ---------- A: Multi-TF “trend” (UP/DOWN/FLAT) ----------
    # Retourne un dict: { '5m':'UP', '15m':'FLAT', ..., 'up': <int>, 'down': <int> }
    def block_A_mtf_trend(self, windows: Dict[str, pd.DataFrame], *, freqs: Tuple[str, ...]) -> Dict[str, Any]:
        A: Dict[str, Any] = {}
        ups = 0
        dns = 0
        thr = self.slope_min

        def _classify_row(row: pd.Series) -> str:
            for k in ("sma20", "sma200", "sma20_slope"):
                if k not in row or pd.isna(row[k]) or not math.isfinite(float(row[k])):
                    return "FLAT"
            sma20 = float(row["sma20"])
            sma200 = float(row["sma200"])
            slope = float(row["sma20_slope"])
            if (sma20 > sma200) and (slope >= thr):  return "UP"
            if (sma20 < sma200) and (slope <= -thr): return "DOWN"
            return "FLAT"

        for f in freqs:
            df = windows.get(f)
            if df is None or df.empty:
                # si tu veux “zéro fallback” total, lève plutôt une erreur :
                # raise ValueError(f"[A[{f}]] fenêtre vide")
                A[f] = "FLAT"
                continue
            # Si tu veux forcer la présence des colonnes (vraiment sans fallback) :
            # _require_cols(df, ["sma20","sma200","sma20_slope"], f"A[{f}]")
            row = df.iloc[-1]
            state = _classify_row(row)
            A[f] = state
            if state == "UP": ups += 1
            elif state == "DOWN": dns += 1

        A["up"] = ups
        A["down"] = dns
        return A
    
    # -------- B: Patterns (5m) ----------
    def block_B_pattern(self, df5m: Optional[pd.DataFrame]) -> Dict[str, bool]:
        """
        Flags (sans fallback, colonnes requises):
          - B1_pullback_up: tendance up (sma20 > sma200, slope>0) + "touch band" vs 20 récemment + recross au-dessus 20
          - B1_pullback_down: symétrique bearish
          - B2_weak_break_up: close > pivot_high * (1 + eps)
          - B2_weak_break_down: close < pivot_low  * (1 - eps)
        eps ~ 0.001 (0.1%)
        """
        out = {
            "B1_pullback_up": False,
            "B1_pullback_down": False,
            "B2_weak_break_up": False,
            "B2_weak_break_down": False,
        }
        if df5m is None or df5m.empty:
            return out

        _require_cols(df5m, ["close", "sma20", "sma200", "sma20_slope", "dist20_atr", "pivot_high", "pivot_low"], "B[5m]")

        c  = df5m["close"].astype(float)
        s20  = df5m["sma20"].astype(float)
        s200 = df5m["sma200"].astype(float)
        slope20 = df5m["sma20_slope"].astype(float)
        d20atr = df5m["dist20_atr"].astype(float)
        piv_hi = df5m["pivot_high"].astype(float)
        piv_lo = df5m["pivot_low"].astype(float)

        if len(df5m) >= max(20, PIVOT_LEN + 2):
            # Trend filters
            trend_up  = bool((s20.iloc[-1] > s200.iloc[-1]) and (slope20.iloc[-1] > +SLOPE_MIN))
            trend_dn  = bool((s20.iloc[-1] < s200.iloc[-1]) and (slope20.iloc[-1] < -SLOPE_MIN))

            # Touch/retest zone vs 20 (sur les N dernières barres)
            look_touch = min(10, len(df5m))
            touched_up = bool((d20atr.iloc[-look_touch:] <= TOUCH_BAND).any())
            # Recrosss 20 à la dernière barre (up)
            recross_up = bool((c.iloc[-2] <= s20.iloc[-2]) and (c.iloc[-1] > s20.iloc[-1]))

            out["B1_pullback_up"] = bool(trend_up and touched_up and recross_up and (d20atr.iloc[-1] <= EXT_MAX_ATR))

            # Down side
            touched_dn = bool((d20atr.iloc[-look_touch:] <= TOUCH_BAND).any())
            recross_dn = bool((c.iloc[-2] >= s20.iloc[-2]) and (c.iloc[-1] < s20.iloc[-1]))

            out["B1_pullback_down"] = bool(trend_dn and touched_dn and recross_dn and (d20atr.iloc[-1] <= EXT_MAX_ATR))

            # Weak break vs mini-pivots (petit dépassement ~0.1%)
            eps = 0.001
            if not np.isnan(piv_hi.iloc[-1]):
                out["B2_weak_break_up"] = bool(c.iloc[-1] > float(piv_hi.iloc[-1]) * (1.0 + eps))
            if not np.isnan(piv_lo.iloc[-1]):
                out["B2_weak_break_down"] = bool(c.iloc[-1] < float(piv_lo.iloc[-1]) * (1.0 - eps))

        return out

    # -------- C: Regime (30m) ----------
    def block_C_regime(self, df30m: Optional[pd.DataFrame], n: int = 20, k: float = 2.0,
                       quiet_thr: float = 0.02, volatile_thr: float = 0.06) -> Dict[str, Any]:
        """
        Regime via bb_width (déjà calculé par prepare_data.add_retail_indicators).
        AUCUN fallback: on exige la colonne.
        """
        if df30m is None or df30m.empty:
            raise ValueError("[C[30m]] DataFrame vide — indicateurs absents")
        _require_cols(df30m, ["bb_width"], "C[30m]")

        bbw = float(df30m["bb_width"].iloc[-1])
        if not np.isfinite(bbw) or bbw <= 0:
            return {"regime": "UNKNOWN", "bb_width": float("nan")}
        if bbw < quiet_thr:
            regime = "QUIET"
        elif bbw > volatile_thr:
            regime = "VOLATILE"
        else:
            regime = "NORMAL"
        return {"regime": regime, "bb_width": bbw}

    # -------- D: RSI circuit breaker (1h) ----------
    def block_D_rsi_circuit(self, df1h: Optional[pd.DataFrame], n: int = 14,
                            overbought: float = 72.0, oversold: float = 28.0) -> Dict[str, Any]:
        """
        AUCUN fallback: on exige la colonne rsi (calculée par prepare_data.add_retail_indicators).
        """
        if df1h is None or df1h.empty:
            raise ValueError("[D[1h]] DataFrame vide — indicateurs absents")
        _require_cols(df1h, ["rsi"], "D[1h]")

        r = float(df1h["rsi"].iloc[-1])
        return {
            "block_long": bool(r > overbought),
            "block_short": bool(r < oversold),
            "rsi": r,
        }

    # ---------- Décision ABCD (NO FALLBACK) ----------
    # Retourne (enter: bool, side: Optional[str], reason: str)
    def abcd_decide(self, A, B, C, D):
        self.last_gate_reason = None  # pas de gate dur en Étape 1

        ups = int(A.get("up", 0))
        dns = int(A.get("down", 0))
        strong_up   = (ups >= 2)
        strong_down = (dns >= 2)
        neutral_A   = (ups == 0 and dns == 0)

        # 1) B1 prioritaire si non contredit fort par A
        if B.get("B1_pullback_up", False) and not strong_down:
            return True, "buy", "B1_PULLBACK"
        if B.get("B1_pullback_down", False) and not strong_up:
            return True, "sell", "B1_PULLBACK"

        # 2) B2 (soft A) : OK si A n’est pas fortement opposé OU si A est neutre
        if B.get("B2_weak_break_up", False) and (not strong_down or neutral_A):
            return True, "buy", "B2_SOFT_A"
        if B.get("B2_weak_break_down", False) and (not strong_up or neutral_A):
            return True, "sell", "B2_SOFT_A"

        # 3) A “classique” (domination claire)
        if ups >= 1 and dns == 0:
            return True, "buy", "A_DOM_UP"
        if dns >= 1 and ups == 0:
            return True, "sell", "A_DOM_DOWN"

        # Rien de déclencheur
        self.last_gate_reason = "ABCD_NO_ENTRY"
        return False, None, "ABCD_NO_ENTRY"
# ---------------------------------------------------------------------
# Optionnel : evaluate() de debug local avec on_trace() — sans fallback
# ---------------------------------------------------------------------
def evaluate(
    df_ctx: Dict[str, pd.DataFrame],
    on_trace: Optional[callable] = None,
    freqs: Iterable[str] = ("5m", "15m", "30m", "1h"),
    bb_n: int = 20, bb_k: float = 2.0,
    bb_quiet: float = 0.02, bb_volatile: float = 0.06,
    rsi_n: int = 14, rsi_overbought: float = 72.0, rsi_oversold: float = 28.0,
) -> Dict[str, Any]:
    """
    Évalue une "bougie" synthétique via les 4 blocs et retourne la décision.
    df_ctx: dict {"5m":df5m, "15m":df15m, "30m":df30m, "1h":df1h}
    AUCUN fallback: si les colonnes requises manquent, on lève une erreur.
    """
    strat = V1BreakoutStrategy(symbol="TEST", frequency="15m")

    # Windows (on prend tels quels)
    windows = {f: df_ctx.get(f) for f in freqs}

    # A
    A = strat.block_A_mtf_trend(windows, freqs=freqs)
    if on_trace:
        on_trace(ts=str(_last(windows.get("15m")).get("timestamp", "NA")) if _last(windows.get("15m")) is not None else "NA",
                 tf="15m", stage="retail", decision="CANDIDATE", reason="A_MTF",
                 ctx={k: A.get(k) for k in ("5m","15m","30m","1h")})

    # B
    B = strat.block_B_pattern(windows.get("5m"))
    if on_trace:
        on_trace(ts="NA", tf="15m", stage="retail", decision="CANDIDATE", reason="B_PATTERN",
                 ctx={k: bool(B.get(k, False)) for k in ["B1_pullback_up","B1_pullback_down","B2_weak_break_up","B2_weak_break_down"]})

    # C
    C = strat.block_C_regime(windows.get("30m"), n=bb_n, k=bb_k, quiet_thr=bb_quiet, volatile_thr=bb_volatile)
    if on_trace:
        on_trace(ts="NA", tf="15m", stage="retail", decision="C_REGIME", reason="C_REGIME",
                 ctx={"regime": C.get("regime"), "bb_width": float(C.get("bb_width", np.nan))})

    # D
    D = strat.block_D_rsi_circuit(windows.get("1h"), n=rsi_n, overbought=rsi_overbought, oversold=rsi_oversold)
    if on_trace and (bool(D.get("block_long", False)) or bool(D.get("block_short", False))):
        on_trace(ts="NA", tf="15m", stage="gate", decision="REJECT", reason="RSI_CUTOFF",
                 ctx={"rsi": float(D.get("rsi", np.nan)),
                      "block_long": bool(D.get("block_long", False)),
                      "block_short": bool(D.get("block_short", False))})

    # Décision
    enter, side, reason = strat.abcd_decide(A, B, C, D)

    if on_trace:
        if not enter or side is None:
            on_trace(ts="NA", tf="15m", stage="gate", decision="REJECT", reason=reason,
                     ctx={"regime": C.get("regime", "UNKNOWN"),
                          "bb_width": float(C.get("bb_width", np.nan)),
                          "rsi": float(D.get("rsi", np.nan))})
        else:
            on_trace(ts="NA", tf="15m", stage="retail", decision="GO", reason="OK",
                     ctx={"regime": C.get("regime", "UNKNOWN"),
                          "bb_width": float(C.get("bb_width", np.nan)),
                          "rsi": float(D.get("rsi", np.nan)),
                          "side": side})

    return {"A": A, "B": B, "C": C, "D": D, "enter": enter, "side": side, "reason": reason}