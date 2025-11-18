#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# eval_router_universal.py — régimes spécifiques, Bayes OFF pour CALM, I/O 100% S3

from __future__ import annotations
import argparse, json, logging, re
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import fsspec

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - [%(name)s] - %(message)s")
logger = logging.getLogger("eval_router_universal")

# Noyau fixe (inchangé)
BASE_FEATURES = ["spread_bps_entry", "quote_churn_10s", "obi_5"]

# Flux gate par régime (ajout)
FLUX_FEATURE_BY_REGIME = {
    "NORMAL":   "aggr_ratio_10s",   # momentum court
    "VOLATILE": "net_delta_15s",    # déséquilibre signé plus robuste
    "CALM":     None,
}

# === Hyperparams Bayésiens par régime ===
BAYES_CFG = {
  "CALM":     {"enable": False, "N0": 10**9, "min_local_n": 10**9, "clip_frac": 0.00},
  "NORMAL":   {"enable": True,  "N0": 300,   "min_local_n": 80,    "clip_frac": 0.07},
  "VOLATILE": {"enable": True,  "N0": 250,   "min_local_n": 80,    "clip_frac": 0.07},
}

# === Utils S3 ===
def s3_read_json(path: str) -> dict:
    with fsspec.open(path, "r") as f:
        return json.load(f)

def s3_write_json(obj: dict, path: str) -> None:
    with fsspec.open(path, "w") as f:
        json.dump(obj, f, indent=2)

def s3_write_csv(df: pd.DataFrame, path: str, index: bool=False) -> None:
    with fsspec.open(path, "w") as f:
        df.to_csv(f, index=index)

def s3_write_text(text: str, path: str) -> None:
    with fsspec.open(path, "w") as f:
        f.write(text)

# === Maths ===
def weighted_mean(v, w) -> float:
    v = pd.to_numeric(v, errors="coerce"); w = pd.to_numeric(w, errors="coerce")
    num = (v*w).sum(min_count=1); den = w.sum(min_count=1)
    return float(num/den) if pd.notna(den) and den!=0 else float("nan")

def bps(x):
    """Convertit en bps. Accepte float/np.float, pd.Series/Index/ndarray."""
    if isinstance(x, (pd.Series, pd.Index, np.ndarray)):
        return pd.to_numeric(x, errors="coerce") * 1e4
    try:
        return float(x) * 1e4
    except Exception:
        return np.nan

# === Parsing combo rr/rf/ts -> colonne PnL ===
def _parse_combo_col(rr: float, rf: int, ts: int, cols: List[str]) -> Optional[str]:
    exact = f"pnl_net_rr{rr:g}_rf{int(rf)}_ts{int(ts)}"
    if exact in cols:
        return exact
    pat = re.compile(r"pnl_net_rr([\d\.]+)_rf(\d+)_ts(\d+)")
    best=None; best_rr=None
    for c in cols:
        m=pat.fullmatch(c)
        if not m: continue
        rr_c=float(m.group(1)); rf_c=int(m.group(2)); ts_c=int(m.group(3))
        if rf_c==rf and ts_c==ts and rr_c>=rr:
            if best is None or rr_c<best_rr:
                best=c; best_rr=rr_c
    if best is None:
        logger.error(f"No matching PnL column for rr>={rr}, rf={rf}, ts={ts}")
    return best

# === Attribution de régime ===
def make_regime_series(df: pd.DataFrame, by: str = "symbol_month",
                       pctl_calm: float = 40.0, pctl_vol: float = 75.0) -> pd.Series:
    if "t" not in df.columns or "atr_pct" not in df.columns:
        raise ValueError("Columns 't' and 'atr_pct' are required")

    atr = pd.to_numeric(df["atr_pct"], errors="coerce")

    def _cut_with_fallback(s: pd.Series) -> pd.Categorical:
        s_num = pd.to_numeric(s, errors="coerce")
        lo = np.nanpercentile(s_num, pctl_calm) if s_num.notna().any() else np.nan
        hi = np.nanpercentile(s_num, pctl_vol)  if s_num.notna().any() else np.nan

        # fallback global ou médian si lo/hi dégénérés
        if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
            g_lo = np.nanpercentile(atr, pctl_calm)
            g_hi = np.nanpercentile(atr, pctl_vol)
            if not np.isfinite(g_lo) or not np.isfinite(g_hi) or g_lo >= g_hi:
                med = np.nanmedian(atr); eps = 1e-12 if np.isfinite(med) else 0.0
                bins = [-np.inf, med - eps, med + eps, np.inf]
            else:
                if abs(g_hi - g_lo) < 1e-12: g_hi += 1e-12
                bins = [-np.inf, g_lo, g_hi, np.inf]
        else:
            if abs(hi - lo) < 1e-12: hi += 1e-12
            bins = [-np.inf, lo, hi, np.inf]

        return pd.cut(s_num, bins, labels=["CALM", "NORMAL", "VOLATILE"])

    if by == "global":
        lo = np.nanpercentile(atr, pctl_calm)
        hi = np.nanpercentile(atr, pctl_vol)
        if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
            med = np.nanmedian(atr); eps = 1e-12 if np.isfinite(med) else 0.0
            bins = [-np.inf, med - eps, med + eps, np.inf]
        else:
            if abs(hi - lo) < 1e-12: hi += 1e-12
            bins = [-np.inf, lo, hi, np.inf]
        return pd.cut(atr, bins, labels=["CALM", "NORMAL", "VOLATILE"])

    if by in ("symbol", "symbol_month") and "symbol" not in df.columns:
        raise KeyError("Column 'symbol' is required for by in {'symbol','symbol_month'}")

    if by == "symbol":
        reg = df.groupby("symbol", observed=False)["atr_pct"].transform(lambda s: _cut_with_fallback(s).astype(object))
        return reg.astype("category")

    # by == "symbol_month"
    # tz-aware -> tz-naive -> Period[M] pour éviter les warnings
    ym = pd.to_datetime(df["t"], utc=True, errors="coerce").dt.tz_convert(None).dt.to_period("M")
    tmp = df.copy(); tmp["__ym__"] = ym
    out = pd.Series(index=df.index, dtype="object")
    for (_, per), idx in tmp.groupby(["symbol", "__ym__"], observed=False).groups.items():
        out.loc[idx] = _cut_with_fallback(atr.loc[idx])
    return out.astype("category")

# === Filtrage micro ===
def apply_feature_rules(df: pd.DataFrame, rules: Dict[str, dict], regime: str,
                        include_flux: bool) -> pd.Series:
    """
    Vote majoritaire par régime.
      - NORMAL/VOLATILE: objectif 2/3 si 3 features présentes ; si 4 présentes (flux inclus), on demande 3/4.
      - CALM: 3/3.
    Seules les features présentes ET avec 'value' définie comptent dans le vote.
    """
    # quelles features pour ce call ?
    feats = list(BASE_FEATURES)
    if include_flux:
        ff = FLUX_FEATURE_BY_REGIME.get(regime)
        if ff: feats.append(ff)

    ok_map = {f: pd.Series(False, index=df.index) for f in feats}
    present_feats = []

    for feat in feats:
        cfg = rules.get(feat, {})
        if feat not in df.columns:
            continue
        thr = cfg.get("value", np.nan)
        if not np.isfinite(thr):
            continue
        val = pd.to_numeric(df[feat], errors="coerce")
        op  = cfg.get("op", "<" if feat!="obi_5" else ">")
        cond = (val < thr) if op == "<" else (val > thr)
        ok_map[feat] = cond.fillna(False)
        present_feats.append(feat)

    if not present_feats:
        return pd.Series(False, index=df.index)

    score = sum(ok_map[f].astype(int) for f in present_feats)

    if regime == "CALM":
        need = len(present_feats)  # tous
    else:
        # 2/3 si 3 feat, 3/4 si 4 feat, sinon “100% des présentes” si <2
        if len(present_feats) >= 4:
            need = 3
        elif len(present_feats) == 3:
            need = 2
        else:
            need = len(present_feats)

    return (score >= need)

def _apply_vote_with_values(dfm_reg: pd.DataFrame, regime: str, values: Dict[str, float], ops: Dict[str, str]) -> pd.Series:
    """
    Applique le vote majoritaire 2/3 (CALM=3/3) avec des 'values' explicites (déjà calculées)
    pour spread_bps_entry, quote_churn_10s, obi_5.
    """
    ok = {}
    for feat in ["spread_bps_entry","quote_churn_10s","obi_5"]:
        if feat not in dfm_reg.columns or feat not in values or not np.isfinite(values[feat]):
            ok[feat] = pd.Series(False, index=dfm_reg.index)
            continue
        v   = pd.to_numeric(dfm_reg[feat], errors="coerce")
        thr = float(values[feat])
        op  = ops.get(feat, "<" if feat!="obi_5" else ">")
        ok[feat] = (v < thr) if op=="<" else (v > thr)

    present = [f for f in ["spread_bps_entry","quote_churn_10s","obi_5"] if f in ok]
    score = sum(ok[f].astype(int) for f in present)
    need  = 3 if regime=="CALM" else 2
    if len(present) < need:  # si moins de 2 features dispo → il faut 100% des présentes
        need = len(present)
    return (score >= need).fillna(False)

def grid_search_quantiles(df_regime, rules_seed, regime_name, grid):
    """
    df_regime: DataFrame filtré sur un régime ("NORMAL" ou "VOLATILE") sur toute l'année 2024.
    rules_seed: dict {feature: {"op":..., "quantile":..., "value":...}} ; on remplace seulement "quantile".
    grid: dict {feature: [list de quantiles à tester]}
    """
    out = []
    if df_regime.empty:
        return pd.DataFrame(columns=["q_spread","q_churn","q_obi","coverage","expectancy_bps","winrate","n","per_month_min_n"])

    # utilitaires
    def qval(series, q):
        s = pd.to_numeric(series, errors="coerce").dropna()
        return float(np.nanquantile(s, q)) if len(s) else np.nan

    # pré-calculs date
    df = df_regime.copy()
    df["year"] = df["t"].dt.year
    df["month"] = df["t"].dt.month
    months = sorted(df["t"].dt.tz_convert(None).dt.to_period("M").unique())

    total_n = int(df["pnl_net_router"].notna().sum())
    if total_n == 0:
        return pd.DataFrame(columns=["q_spread","q_churn","q_obi","coverage","expectancy_bps","winrate","n","per_month_min_n"])

    # boucles sur la grille
    for q_spread in grid["spread_bps_entry"]:
        for q_churn in grid["quote_churn_10s"]:
            for q_obi in grid["obi_5"]:
                # clone règles seed
                rules = json.loads(json.dumps(rules_seed))
                rules["spread_bps_entry"]["quantile"] = q_spread
                rules["quote_churn_10s"]["quantile"]  = q_churn
                rules["obi_5"]["quantile"]            = q_obi

                # thresholds à partir de ces quantiles
                thr = {
                    "spread_bps_entry": qval(df["spread_bps_entry"], q_spread),
                    "quote_churn_10s":  qval(df["quote_churn_10s"],  q_churn),
                    "obi_5":            qval(df["obi_5"],            q_obi),
                }
                # si un threshold est NaN → combo invalide
                if not all(np.isfinite(list(thr.values()))):
                    continue

                # applique vote 2/3
                mask_spread = (pd.to_numeric(df["spread_bps_entry"], errors="coerce") <
                               thr["spread_bps_entry"]) if rules["spread_bps_entry"]["op"] == "<" else \
                              (pd.to_numeric(df["spread_bps_entry"], errors="coerce") >
                               thr["spread_bps_entry"])
                mask_churn  = (pd.to_numeric(df["quote_churn_10s"], errors="coerce") <
                               thr["quote_churn_10s"]) if rules["quote_churn_10s"]["op"] == "<" else \
                              (pd.to_numeric(df["quote_churn_10s"], errors="coerce") >
                               thr["quote_churn_10s"])
                mask_obi    = (pd.to_numeric(df["obi_5"], errors="coerce") >
                               thr["obi_5"]) if rules["obi_5"]["op"] == ">" else \
                              (pd.to_numeric(df["obi_5"], errors="coerce") <
                               thr["obi_5"])

                score = mask_spread.astype(int) + mask_churn.astype(int) + mask_obi.astype(int)
                sel = (score >= 2)  # 2/3 pour NORMAL/VOLATILE

                # agrégation mois → stabilité
                rows = []
                for per in months:
                    y, m = int(per.year), int(per.month)
                    idx = (df["year"]==y) & (df["month"]==m)
                    if not idx.any():
                        continue
                    pnl = pd.to_numeric(df.loc[idx & sel, "pnl_net_router"], errors="coerce").dropna()
                    if len(pnl)==0:
                        continue
                    rows.append({
                        "n": len(pnl),
                        "e": float(pnl.mean()),
                        "w": float((pnl>0).mean())
                    })
                if not rows:
                    continue

                months_df = pd.DataFrame(rows)
                n = int(months_df["n"].sum())
                coverage = n / total_n
                # moyennes pondérées par n (puis conversion bps)
                exp_bps = float((months_df["e"]*months_df["n"]).sum() / months_df["n"].sum() * 1e4)
                wr      = float((months_df["w"]*months_df["n"]).sum() / months_df["n"].sum())
                per_month_min_n = int(months_df["n"].min())

                out.append({
                    "q_spread": q_spread, "q_churn": q_churn, "q_obi": q_obi,
                    "coverage": coverage, "expectancy_bps": exp_bps, "winrate": wr,
                    "n": n, "per_month_min_n": per_month_min_n
                })

    res = pd.DataFrame(out).sort_values(["expectancy_bps","winrate","per_month_min_n","coverage"], ascending=[False,False,False,True])
    return res

def compute_quantile(series: pd.Series, q: float)->float:
    s=pd.to_numeric(series,errors="coerce").dropna()
    return float(np.nanquantile(s,q)) if len(s) else float("nan")

def bayes_blend(prior: float, local: float, n_local: int, n0: int)->float:
    if not np.isfinite(local):
        return prior
    alpha = n_local / (n_local + n0) if (n_local+n0)>0 else 0.0
    return float((1-alpha)*prior + alpha*local)

# === Runner ===
def run_eval(labels_path, router_params_json, feature_cutoffs_prior_json, out_prefix,
             mode="both", symbol=None, by="symbol_month", pctl_calm=40.0, pctl_vol=75.0,
             grid_search: bool=False, args=None):

    # === Chargement labels ===
    df = pd.read_csv(labels_path)
    if "t" not in df.columns:
        raise SystemExit("CSV needs 't'")
    df["t"] = pd.to_datetime(df["t"], utc=True, errors="coerce")
    df = df.dropna(subset=["t"]).sort_values("t").copy()

    # === Déduction symbole si absent ===
    if symbol is None:
        if "symbol" in df.columns and df["symbol"].nunique() == 1:
            symbol = str(df["symbol"].iloc[0])
        else:
            m = re.search(r"/report/([^/]+)/", labels_path)
            symbol = m.group(1) if m else "UNKNOWN"

    # === Colonnes PnL
    pnl_cols = [c for c in df.columns if c.startswith("pnl_net_rr")]
    if not pnl_cols:
        raise SystemExit("No 'pnl_net_rr*_rf*_ts*' columns found in labels.")

    # === Régimes
    df["regime_eval"] = make_regime_series(df, by, pctl_calm, pctl_vol).astype(str).str.upper()

    # === Params / Priors JSON
    params_all = s3_read_json(router_params_json)
    prior_all  = s3_read_json(feature_cutoffs_prior_json)

    if symbol not in params_all:
        raise SystemExit(f"Symbol '{symbol}' not found in router_params_json.")
    if symbol not in prior_all:
        raise SystemExit(f"Symbol '{symbol}' not found in feature_cutoffs_prior_json.")

    sym_params = params_all[symbol]

    # === Choix combo PnL par régime
    chosen = {}
    for reg in ["CALM", "NORMAL", "VOLATILE"]:
        if reg not in sym_params:
            continue
        rr = float(sym_params[reg]["rr_min"])
        rf = int(sym_params[reg]["risk_floor_bps"])
        ts = int(sym_params[reg]["time_stop"])
        col = _parse_combo_col(rr, rf, ts, pnl_cols)
        if not col:
            raise SystemExit(f"No PnL column for regime={reg} (rr>={rr}, rf={rf}, ts={ts})")
        chosen[reg] = col

    # === Assembler la colonne PnL du router
    df["pnl_net_router"] = np.nan
    for reg, col in chosen.items():
        sel = df["regime_eval"].eq(reg)
        df.loc[sel, "pnl_net_router"] = pd.to_numeric(df.loc[sel, col], errors="coerce")

    # === Init règles FIXED / ROLLING à partir du prior JSON
    def init_rules(reg: str) -> Dict[str, dict]:
        rules = {}
        block = prior_all[symbol].get(reg, {})
        # noyau fixe
        for feat in BASE_FEATURES:
            meta = block.get(feat, {})
            op   = meta.get("op", "<" if feat!="obi_5" else ">")
            qntl = meta.get("quantile", np.nan)
            val  = meta.get("value", np.nan)
            rules[feat] = {
                "op": op,
                "quantile": float(qntl) if qntl is not None else np.nan,
                "value": float(val) if val is not None else np.nan
            }
        # flux optionnel
        flux_feat = FLUX_FEATURE_BY_REGIME.get(reg)
        if flux_feat:
            meta = block.get(flux_feat, {})
            op   = meta.get("op", ">")  # flux = plus grand est meilleur
            qntl = meta.get("quantile", np.nan)
            val  = meta.get("value", np.nan)
            rules[flux_feat] = {
                "op": op,
                "quantile": float(qntl) if qntl is not None else np.nan,
                "value": float(val) if val is not None else np.nan
            }
        return rules

    rules_fixed = {reg: init_rules(reg) for reg in ["CALM", "NORMAL", "VOLATILE"]}

    # log si value manquante en FIXE
    for reg, feats in rules_fixed.items():
        for feat, cfg in feats.items():
            if not np.isfinite(cfg.get("value", np.nan)):
                logger.warning(f"[FIXED] {symbol}/{reg}/{feat}: 'value' absente dans le JSON -> la feature ne comptera pas en FIXE.")

    # Copie pour ROLLING
    rules_rolling = json.loads(json.dumps(rules_fixed))

    # --- GRID des quantiles flux (NORMAL/VOLATILE) : LARGE SWEEP ---
        # --- GRID des quantiles flux (CALM/NORMAL/VOLATILE) : optionnel ---
    if getattr(args, "grid_search", False):
        # ⚠️ Assure-toi que ce mapping existe en amont du fichier:
        # FLUX_FEATURE_BY_REGIME = {
        #     "CALM":     "aggr_ratio_10s",   # flux précurseur en marché calme
        #     "NORMAL":   "aggr_ratio_10s",   # momentum court
        #     "VOLATILE": "net_delta_15s",    # persistance directionnelle
        # }

        def _try_q(reg: str, feat: str | None, q_list):
            """Teste une grille de quantiles pour la feature flux du régime 'reg'."""
            if not feat:
                return None
            g = df[df["regime_eval"].eq(reg)].copy()
            if g.empty or feat not in g.columns:
                return None

            # On part des règles FIXES actuelles (on ne touche qu'au quantile+value du flux)
            base_rules = json.loads(json.dumps(rules_fixed))
            rows = []

            for q in q_list:
                test_rules = json.loads(json.dumps(base_rules))
                test_rules[reg].setdefault(feat, {"op": ">"})
                test_rules[reg][feat]["quantile"] = float(q)
                test_rules[reg][feat]["value"]    = compute_quantile(g[feat], q)

                mask = apply_feature_rules(g, test_rules[reg], regime=reg, include_flux=True)
                n_reg = int(len(g))
                n_sel = int(mask.sum())
                cov   = (n_sel / n_reg) if n_reg else np.nan

                pnl   = pd.to_numeric(g["pnl_net_router"], errors="coerce")
                pnl_f = pnl[mask]
                exp   = float(pnl_f.mean()) if n_sel else np.nan
                wr    = float((pnl_f > 0).mean()) if n_sel else np.nan

                rows.append({
                    "regime": reg,
                    "flux_feat": feat,
                    "q_flux": float(q),
                    "coverage": cov,
                    "expectancy": exp,
                    "winrate": wr,
                    "n": n_sel
                })

            return pd.DataFrame(rows)

        # Grilles larges par régime (CALM un peu plus “grossière”)
        grid_calm_q    = list(np.round(np.arange(0.20, 0.86, 0.05), 3))   # 0.20 → 0.85 step 0.05
        grid_normal_q  = list(np.round(np.arange(0.55, 0.96, 0.02), 3))   # 0.55 → 0.95 step 0.02
        grid_vola_q    = list(np.round(np.arange(0.60, 0.96, 0.02), 3))   # 0.60 → 0.95 step 0.02

        # Lancements
        dfC = _try_q("CALM",     FLUX_FEATURE_BY_REGIME.get("CALM"),     grid_calm_q)
        dfN = _try_q("NORMAL",   FLUX_FEATURE_BY_REGIME.get("NORMAL"),   grid_normal_q)
        dfV = _try_q("VOLATILE", FLUX_FEATURE_BY_REGIME.get("VOLATILE"), grid_vola_q)

        # Sélection par régime (tunnels de coverage & min n)
        def pick_top_by_regime(reg: str, df_res: pd.DataFrame | None):
            if df_res is None or df_res.empty:
                return df_res
            if reg == "CALM":
                cov_lo, cov_hi, min_n = 0.05, 0.15, 5
            else:
                cov_lo, cov_hi, min_n = 0.08, 0.20, 5
            sel = df_res["coverage"].between(cov_lo, cov_hi) & (df_res["n"] >= min_n)
            return df_res.loc[sel].sort_values(
                ["expectancy", "winrate", "n"],
                ascending=[False, False, False]
            )

        topC = pick_top_by_regime("CALM",     dfC) if dfC is not None else None
        topN = pick_top_by_regime("NORMAL",   dfN) if dfN is not None else None
        topV = pick_top_by_regime("VOLATILE", dfV) if dfV is not None else None

        # Sauvegardes CSV
        if dfC is not None: s3_write_csv(dfC, f"{out_prefix}_grid_flux_CALM_all.csv", index=False)
        if dfN is not None: s3_write_csv(dfN, f"{out_prefix}_grid_flux_NORMAL_all.csv", index=False)
        if dfV is not None: s3_write_csv(dfV, f"{out_prefix}_grid_flux_VOLATILE_all.csv", index=False)

        if topC is not None: s3_write_csv(topC.head(20), f"{out_prefix}_grid_flux_CALM_top.csv", index=False)
        if topN is not None: s3_write_csv(topN.head(20), f"{out_prefix}_grid_flux_NORMAL_top.csv", index=False)
        if topV is not None: s3_write_csv(topV.head(20), f"{out_prefix}_grid_flux_VOLATILE_top.csv", index=False)

        # Résumé texte
        def _fmt_block(name: str, df_top: pd.DataFrame | None):
            lines = [f"=== {symbol} / {name} — top {0 if (df_top is None or df_top.empty) else min(20, len(df_top))} ==="]
            if df_top is None or df_top.empty:
                lines.append("  (aucun combo éligible)")
                return "\n".join(lines)
            for _, r in df_top.head(20).iterrows():
                lines.append(
                    f"  q_flux={r.q_flux:.3f}  |  cov={r.coverage:.1%}  "
                    f"exp={r.expectancy:.4f}  WR={r.winrate:.1%}  n={int(r.n)}"
                )
            return "\n".join(lines)

        summary = "\n\n".join([
            _fmt_block("CALM",     topC),
            _fmt_block("NORMAL",   topN),
            _fmt_block("VOLATILE", topV),
        ])
        s3_write_text(summary, f"{out_prefix}_grid_flux_summary.txt")
        logger.info("Grid-search flux terminé (CALM/NORMAL/VOLATILE).")
        return  # on s'arrête ici (pas de rolling/bayes dans ce mode)
    
    # === Découpage mensuel
    df["year"]  = df["t"].dt.year
    df["month"] = df["t"].dt.month
    months = sorted(df["t"].dt.tz_convert(None).dt.to_period("M").unique())

    base_rows, filt_rows, table_rows, diag_rows = [], [], [], []

    def agg(dfx: pd.DataFrame):
        pnl = pd.to_numeric(dfx["pnl_net_router"], errors="coerce").dropna()
        n = int(len(pnl))
        e = float(pnl.mean()) if n else float("nan")
        w = float((pnl > 0).mean()) if n else float("nan")
        return n, e, w

    for per in months:
        y, m = int(per.year), int(per.month)
        dfm = df[(df["year"] == y) & (df["month"] == m)].copy()
        if dfm.empty:
            continue

        # Baseline
        nb, eb, wb = agg(dfm)
        for _, r in dfm.iterrows():
            base_rows.append({
                "t": r["t"], "symbol": symbol, "regime_eval": r["regime_eval"],
                "pnl_net_router": r["pnl_net_router"], "year": y, "month": m
            })

        # Masques (no-flux vs flux)
        masks_fixed_no_flux = pd.Series(False, index=dfm.index)
        masks_roll_no_flux  = pd.Series(False, index=dfm.index)
        masks_fixed_flux    = pd.Series(False, index=dfm.index)
        masks_roll_flux     = pd.Series(False, index=dfm.index)

        for reg in ["CALM","NORMAL","VOLATILE"]:
            sel = dfm["regime_eval"].eq(reg)
            if not sel.any():
                continue

            # sans flux
            masks_fixed_no_flux.loc[sel] = apply_feature_rules(
                dfm.loc[sel], rules_fixed[reg], regime=reg, include_flux=False).values
            masks_roll_no_flux.loc[sel]  = apply_feature_rules(
                dfm.loc[sel], rules_rolling[reg], regime=reg, include_flux=False).values

            # avec flux
            masks_fixed_flux.loc[sel] = apply_feature_rules(
                dfm.loc[sel], rules_fixed[reg], regime=reg, include_flux=True).values
            masks_roll_flux.loc[sel]  = apply_feature_rules(
                dfm.loc[sel], rules_rolling[reg], regime=reg, include_flux=True).values

            # --- Diagnostic score noyau (3 feats) en FIXE
            sub = dfm.loc[sel, BASE_FEATURES].copy()
            rules_reg = rules_fixed[reg]
            score = pd.Series(0, index=sub.index, dtype=int)
            for feat in BASE_FEATURES:
                if feat not in sub.columns:
                    continue
                val = pd.to_numeric(sub[feat], errors="coerce")
                cfg = rules_reg.get(feat, {})
                thr = cfg.get("value", np.nan)
                op  = cfg.get("op", "<" if feat != "obi_5" else ">")
                if not np.isfinite(thr):
                    continue
                cond = (val < thr) if op == "<" else (val > thr)
                score += cond.astype(int)
            dist = score.value_counts(normalize=True).sort_index()
            logger.info(f"[{symbol}/{reg}] Score noyau (0..3): " + ", ".join(f"{k}:{v:.2%}" for k, v in dist.items()))

        # --- Stats par régime (no-flux vs flux)
        for reg in ["CALM", "NORMAL", "VOLATILE"]:
            sel = dfm["regime_eval"].eq(reg)
            if not sel.any():
                continue
            nb_reg = int(sel.sum())

            # FIXE no-flux
            sel_fix0 = sel & masks_fixed_no_flux
            nf0_reg  = int(sel_fix0.sum())
            pnl_fix0 = pd.to_numeric(dfm.loc[sel_fix0, "pnl_net_router"], errors="coerce")
            e_fix0   = float(pnl_fix0.mean()) if nf0_reg else float("nan")
            w_fix0   = float((pnl_fix0 > 0).mean()) if nf0_reg else float("nan")

            # ROLLING no-flux
            sel_roll0 = sel & masks_roll_no_flux
            nr0_reg   = int(sel_roll0.sum())
            pnl_roll0 = pd.to_numeric(dfm.loc[sel_roll0, "pnl_net_router"], errors="coerce")
            e_roll0   = float(pnl_roll0.mean()) if nr0_reg else float("nan")
            w_roll0   = float((pnl_roll0 > 0).mean()) if nr0_reg else float("nan")

            # FIXE flux
            sel_fix1 = sel & masks_fixed_flux
            nf1_reg  = int(sel_fix1.sum())
            pnl_fix1 = pd.to_numeric(dfm.loc[sel_fix1, "pnl_net_router"], errors="coerce")
            e_fix1   = float(pnl_fix1.mean()) if nf1_reg else float("nan")
            w_fix1   = float((pnl_fix1 > 0).mean()) if nf1_reg else float("nan")

            # ROLLING flux
            sel_roll1 = sel & masks_roll_flux
            nr1_reg   = int(sel_roll1.sum())
            pnl_roll1 = pd.to_numeric(dfm.loc[sel_roll1, "pnl_net_router"], errors="coerce")
            e_roll1   = float(pnl_roll1.mean()) if nr1_reg else float("nan")
            w_roll1   = float((pnl_roll1 > 0).mean()) if nr1_reg else float("nan")

            # log diag + seuils utilisés ce mois-ci
            diag_rows.append({
                "symbol": symbol, "year": y, "month": m, "regime": reg,
                "n_base_regime": nb_reg,
                "n_fixed_regime_noflux": nf0_reg, "coverage_fixed_regime_noflux": (nf0_reg/nb_reg) if nb_reg else np.nan,
                "expectancy_fixed_noflux": e_fix0, "winrate_fixed_noflux": w_fix0,
                "n_rolling_regime_noflux": nr0_reg, "coverage_rolling_regime_noflux": (nr0_reg/nb_reg) if nb_reg else np.nan,
                "expectancy_rolling_noflux": e_roll0, "winrate_rolling_noflux": w_roll0,

                "n_fixed_regime_flux": nf1_reg, "coverage_fixed_regime_flux": (nf1_reg/nb_reg) if nb_reg else np.nan,
                "expectancy_fixed_flux": e_fix1, "winrate_fixed_flux": w_fix1,
                "n_rolling_regime_flux": nr1_reg, "coverage_rolling_regime_flux": (nr1_reg/nb_reg) if nb_reg else np.nan,
                "expectancy_rolling_flux": e_roll1, "winrate_rolling_flux": w_roll1,

                # seuils (FIXE + ROLLING) — on dump toutes les keys présentes
                **{f"{feat}_thr_fix":  rules_fixed[reg].get(feat, {}).get("value", np.nan)   for feat in rules_fixed[reg].keys()},
                **{f"{feat}_thr_roll": rules_rolling[reg].get(feat, {}).get("value", np.nan) for feat in rules_rolling[reg].keys()},
            })

        # Tables mensuelles globales (les 4 variantes)
        df_fix0 = dfm[masks_fixed_no_flux].copy()
        df_roll0 = dfm[masks_roll_no_flux].copy()
        df_fix1 = dfm[masks_fixed_flux].copy()
        df_roll1 = dfm[masks_roll_flux].copy()

        nf0, ef0, wf0 = agg(df_fix0);  nr0, er0, wr0 = agg(df_roll0)
        nf1, ef1, wf1 = agg(df_fix1);  nr1, er1, wr1 = agg(df_roll1)

        table_rows.append({
            "symbol": symbol, "year": y, "month": m,
            "n_base": nb, "expectancy_base": eb, "winrate_base": wb,

            "n_filt_fixed_noflux": nf0, "expectancy_filt_fixed_noflux": ef0, "winrate_filt_fixed_noflux": wf0,
            "n_filt_rolling_noflux": nr0, "expectancy_filt_rolling_noflux": er0, "winrate_filt_rolling_noflux": wr0,
            "coverage_fixed_noflux": (nf0/nb if nb else np.nan),
            "coverage_rolling_noflux": (nr0/nb if nb else np.nan),

            "n_filt_fixed_flux": nf1, "expectancy_filt_fixed_flux": ef1, "winrate_filt_fixed_flux": wf1,
            "n_filt_rolling_flux": nr1, "expectancy_filt_rolling_flux": er1, "winrate_filt_rolling_flux": wr1,
            "coverage_fixed_flux": (nf1/nb if nb else np.nan),
            "coverage_rolling_flux": (nr1/nb if nb else np.nan),
        })

        # Lignes filtrées (4 modes explicites)
        for _, r in df_fix0.iterrows():
            filt_rows.append({
                "t": r["t"], "symbol": symbol, "regime_eval": r["regime_eval"],
                "pnl_net_router": r["pnl_net_router"], "mode": "fixed_noflux", "year": y, "month": m
            })
        for _, r in df_roll0.iterrows():
            filt_rows.append({
                "t": r["t"], "symbol": symbol, "regime_eval": r["regime_eval"],
                "pnl_net_router": r["pnl_net_router"], "mode": "rolling_noflux", "year": y, "month": m
            })
        for _, r in df_fix1.iterrows():
            filt_rows.append({
                "t": r["t"], "symbol": symbol, "regime_eval": r["regime_eval"],
                "pnl_net_router": r["pnl_net_router"], "mode": "fixed_flux", "year": y, "month": m
            })
        for _, r in df_roll1.iterrows():
            filt_rows.append({
                "t": r["t"], "symbol": symbol, "regime_eval": r["regime_eval"],
                "pnl_net_router": r["pnl_net_router"], "mode": "rolling_flux", "year": y, "month": m
            })

        # === Mise à jour Bayésienne (prépare M+1) ===
        if mode in ("rolling", "both"):
            for reg in ["CALM", "NORMAL", "VOLATILE"]:
                cfg_reg = BAYES_CFG[reg]
                if not cfg_reg["enable"]:
                    continue

                dfr  = dfm[dfm["regime_eval"].eq(reg)]
                n0   = int(cfg_reg["N0"])
                nmin = int(cfg_reg["min_local_n"])
                clip = float(cfg_reg["clip_frac"])

                for feat, cfg in rules_rolling[reg].items():
                    q = cfg.get("quantile", np.nan)
                    if not np.isfinite(q):
                        continue
                    prior   = float(cfg.get("value", np.nan))
                    local   = compute_quantile(dfr[feat], q)
                    n_local = int(dfr[feat].notna().sum())

                    if n_local < nmin or not np.isfinite(prior):
                        post = prior
                    else:
                        post = bayes_blend(prior, local, n_local, n0)
                        if np.isfinite(prior) and prior != 0 and np.isfinite(post):
                            lo = prior * (1.0 - clip); hi = prior * (1.0 + clip)
                            post = float(np.clip(post, min(lo, hi), max(lo, hi)))

                        op = cfg.get("op", "<" if feat != "obi_5" else ">")
                        if np.isfinite(prior) and np.isfinite(local) and n_local >= n0:
                            if op == "<":
                                post = min(post, prior)
                            elif op == ">":
                                post = max(post, prior)

                    cfg["value"] = float(post) if np.isfinite(post) else prior

    # === Sorties S3 ===
    base_out = f"{out_prefix}_rolling_baseline.csv"
    filt_out = f"{out_prefix}_rolling_filtered.csv"
    tbl_out  = f"{out_prefix}_rolling_summary_table.csv"
    diag_out = f"{out_prefix}_rolling_diag_by_regime.csv"
    post_out = f"{out_prefix}_rolling_feature_cutoffs_POST.json"
    txt_out  = f"{out_prefix}_rolling_summary.txt"

    s3_write_csv(pd.DataFrame(base_rows), base_out, index=False)
    s3_write_csv(pd.DataFrame(filt_rows), filt_out, index=False)
    s3_write_csv(pd.DataFrame(table_rows).sort_values(["year","month"]), tbl_out, index=False)
    s3_write_csv(pd.DataFrame(diag_rows).sort_values(["year","month","regime"]), diag_out, index=False)

    if mode in ("rolling", "both"):
        s3_write_json({symbol: rules_rolling}, post_out)

    # === Récap agrégé (en bps)
    table_df = pd.DataFrame(table_rows)

    def aggG(sub, nc, ec, wc):
        n = int(pd.to_numeric(sub[nc], errors="coerce").sum())
        e = weighted_mean(pd.to_numeric(sub[ec], errors="coerce") * 1e4, sub[nc])  # expectancy en bps
        w = weighted_mean(pd.to_numeric(sub[wc], errors="coerce"), sub[nc])
        return n, e, w

    nb, eb_bps, wb = aggG(table_df, "n_base", "expectancy_base", "winrate_base")

    nf0, ef0_bps, wf0 = aggG(table_df, "n_filt_fixed_noflux",   "expectancy_filt_fixed_noflux",   "winrate_filt_fixed_noflux")
    nr0, er0_bps, wr0 = aggG(table_df, "n_filt_rolling_noflux", "expectancy_filt_rolling_noflux", "winrate_filt_rolling_noflux")

    nf1, ef1_bps, wf1 = aggG(table_df, "n_filt_fixed_flux",     "expectancy_filt_fixed_flux",     "winrate_filt_fixed_flux")
    nr1, er1_bps, wr1 = aggG(table_df, "n_filt_rolling_flux",   "expectancy_filt_rolling_flux",   "winrate_filt_rolling_flux")

    recap = "\n".join([
        f"==== RÉCAP (UNIVERSAL) {symbol} ====",
        "Baseline (aucun filtre)",
        f"  • Trades: {nb}",
        (f"  • Expectancy: {eb_bps:.2f} bps/trade" if np.isfinite(eb_bps) else "  • Expectancy: NaN"),
        (f"  • Winrate: {wb:.1%}" if np.isfinite(wb) else "  • Winrate: NaN"),
        "Filtres micro — FIXE (sans flux)",
        (f"  • Trades: {nf0} (coverage {nf0/nb:.1%})" if nb else f"  • Trades: {nf0}"),
        (f"  • Expectancy: {ef0_bps:.2f} bps/trade" if np.isfinite(ef0_bps) else "  • Expectancy: NaN"),
        (f"  • Winrate: {wf0:.1%}" if np.isfinite(wf0) else "  • Winrate: NaN"),
        "Filtres micro — FIXE (avec flux)",
        (f"  • Trades: {nf1} (coverage {nf1/nb:.1%})" if nb else f"  • Trades: {nf1}"),
        (f"  • Expectancy: {ef1_bps:.2f} bps/trade" if np.isfinite(ef1_bps) else "  • Expectancy: NaN"),
        (f"  • Winrate: {wf1:.1%}" if np.isfinite(wf1) else "  • Winrate: NaN"),
        "Filtres micro — ROLLING (sans flux)",
        (f"  • Trades: {nr0} (coverage {nr0/nb:.1%})" if nb else f"  • Trades: {nr0}"),
        (f"  • Expectancy: {er0_bps:.2f} bps/trade" if np.isfinite(er0_bps) else "  • Expectancy: NaN"),
        (f"  • Winrate: {wr0:.1%}" if np.isfinite(wr0) else "  • Winrate: NaN"),
        "Filtres micro — ROLLING (avec flux)",
        (f"  • Trades: {nr1} (coverage {nr1/nb:.1%})" if nb else f"  • Trades: {nr1}"),
        (f"  • Expectancy: {er1_bps:.2f} bps/trade" if np.isfinite(er1_bps) else "  • Expectancy: NaN"),
        (f"  • Winrate: {wr1:.1%}" if np.isfinite(wr1) else "  • Winrate: NaN"),
        "",
        f"(Fichiers: baseline={base_out} | filtered={filt_out} | table={tbl_out} | diag={diag_out} | post={post_out})"
    ])
    s3_write_text(recap, txt_out)

# === CLI ===
def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--labels", required=True)
    p.add_argument("--router-params-json", required=True)  # s3://.../router_params_by_regime-2023.json
    p.add_argument("--feature-cutoffs-prior", required=True)  # s3://.../router_feature_cutoffs_by_regime-2023_values.json
    p.add_argument("--out-prefix", required=True)  # s3://.../BTCUSDT_2024_universal
    p.add_argument("--symbol", default=None)
    p.add_argument("--mode", choices=["fixed","rolling","both"], default="both")
    p.add_argument("--by", choices=["global","symbol","symbol_month"], default="symbol_month")
    p.add_argument("--pctl-calm", type=float, default=40.0)
    p.add_argument("--pctl-volatile", type=float, default=75.0)
    p.add_argument("--grid-search", action="store_true",
               help="Teste une grille de quantiles (mode fixed) et écrit les tops en S3, puis s'arrête.")
    return p.parse_args()

def main():
    a=parse_args()
    run_eval(a.labels, a.router_params_json, a.feature_cutoffs_prior, a.out_prefix,
             mode=a.mode, symbol=a.symbol, by=a.by, pctl_calm=a.pctl_calm, pctl_vol=a.pctl_volatile,
             grid_search=a.grid_search, args=a)

if __name__=="__main__":
    main()