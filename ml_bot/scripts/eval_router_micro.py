#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_router_micro.py
--------------------
Évalue la perf du routeur à partir du CSV multigrille enrichi (micro+PnL),
sans relire L1/BOOK : on sélectionne pour chaque trade la colonne PnL
correspondant à la combo choisie par le routeur + on applique les filtres micro.

Entrées:
- labels multigrille:   CSV (ex: 2023_signals_multigrid_micro.csv)
- router params:        JSON (router_params_by_regime.json)
- feature cutoffs:      JSON (router_feature_cutoffs.json)

Sorties:
- eval_micro_baseline.csv   (toutes les lignes, sans filtre micro)
- eval_micro_filtered.csv   (après filtres micro)
- eval_micro_summary.txt    (KPIs agrégés baseline vs filtré)

Notes:
- Régimes reconstruits par percentiles d'atr_pct (même logique que synthèse).
- Mode strict : aucun fallback DEFAULT. Si (symbol, regime) absent → on n'évalue pas ces lignes.
- Filtres micro stricts par (symbol, regime) (fail-closed).
- Expectancy = mean(pnl_net) ; n = nombre de trades retenus.
"""

from __future__ import annotations
import argparse, json, logging, re
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import fsspec

# ---------------- Logging ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(name)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("eval_router_micro")

# ---------------- S3 utils ----------------
def _normalize_url(url: str) -> str:
    return re.sub(r"^([A-Za-z0-9]+)://", lambda m: m.group(1).lower() + "://", url)

def read_csv_any(path: str, storage_options: dict | None = None) -> pd.DataFrame:
    p = _normalize_url(path)
    with fsspec.open(p, "r", **(storage_options or {})) as f:
        return pd.read_csv(f)

def write_csv_any(df: pd.DataFrame, path: str, storage_options: dict | None = None) -> None:
    p = _normalize_url(path)
    with fsspec.open(p, "w", **(storage_options or {})) as f:
        df.to_csv(f, index=False)

def read_json_any(path: str, storage_options: dict | None = None) -> dict:
    p = _normalize_url(path)
    with fsspec.open(p, "r", **(storage_options or {})) as f:
        return json.load(f)

def write_text_any(text: str, path: str, storage_options: dict | None = None) -> None:
    p = _normalize_url(path)
    with fsspec.open(p, "w", **(storage_options or {})) as f:
        f.write(text)

def _open_json_s3_strict(path: str, storage_options: dict | None = None) -> dict | None:
    """Lit un JSON via fsspec (S3). Si erreur, renvoie None (on n'évalue pas)."""
    try:
        p = _normalize_url(path)
        with fsspec.open(p, "r", **(storage_options or {})) as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"Missing JSON: {path}")
        return None
    except Exception as e:
        logger.error(f"Cannot read JSON '{path}': {e}")
        return None

def _write_empty_outputs(out_baseline_csv: str, out_filtered_csv: str, out_summary_txt: str, storage_options: dict | None = None) -> None:
    empty_base = pd.DataFrame(columns=["t","symbol","side","entry","atr_pct","regime_eval","chosen_pnl_col","pnl_net_router","kept_by_micro"])
    write_csv_any(empty_base, out_baseline_csv, storage_options=storage_options)
    write_csv_any(empty_base, out_filtered_csv, storage_options=storage_options)
    write_text_any("No trading: required JSON config missing (router params and/or feature cutoffs).\n",
                   out_summary_txt, storage_options=storage_options)

# ---------------- Helpers ----------------
def _combo_to_col(rr: float, rf: int, ts: int) -> str:
    return f"pnl_net_rr{rr:.2f}_rf{int(rf)}_ts{int(ts)}"

def _detect_all_combo_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if re.fullmatch(r"pnl_net_rr[\d\.]+_rf\d+_ts\d+", c)]

def _router_and_combo_diagnostics(
    df: pd.DataFrame,
    router_params: Dict[str, Any],
    pnl_cols: List[str],
) -> pd.DataFrame:
    """
    Pour chaque (symbol, regime_eval) présent dans df:
      - has_router_params: 1/0
      - chosen_col: nom de colonne attendu (ou "")
      - combo_exists: 1/0 (colonne réellement présente)
    """
    rows = []
    pnl_set = set(pnl_cols)

    for (sym, reg), g in df.groupby(["symbol","regime_eval"], observed=False, sort=False):
        sym, reg = str(sym), str(reg)
        rp = router_params.get(sym, {}).get(reg)
        has_rp = int(isinstance(rp, dict))
        col = ""
        combo_exists = 0
        rr = rf = ts = None

        if has_rp:
            try:
                rr = float(rp["rr_min"]); rf = int(rp["risk_floor_bps"]); ts = int(rp["time_stop"])
                col = _combo_to_col(rr, rf, ts)
                combo_exists = int(col in pnl_set)
            except Exception:
                has_rp = 0  # invalide

        rows.append(dict(
            symbol=sym, regime=reg, n_rows=len(g),
            has_router_params=has_rp,
            rr_min=rr, risk_floor_bps=rf, time_stop=ts,
            chosen_col=col, combo_exists=combo_exists
        ))
    return pd.DataFrame(rows).sort_values(["symbol","regime"])

def _build_regime_series(df: pd.DataFrame, by: str, pctl_calm: float, pctl_vol: float) -> pd.Series:
    assert "atr_pct" in df.columns, "labels CSV doit contenir 'atr_pct'."
    d = df.copy()
    d["t"] = pd.to_datetime(d["t"], utc=True, errors="coerce")
    s = d["atr_pct"].astype(float)

    if by == "global":
        p_low  = np.nanpercentile(s, pctl_calm)
        p_high = np.nanpercentile(s, pctl_vol)
        return pd.cut(s, [-np.inf, p_low, p_high, np.inf], labels=["CALM","NORMAL","VOLATILE"])

    if by == "symbol":
        def lab(g: pd.DataFrame) -> pd.Series:
            ss = g["atr_pct"].astype(float)
            if ss.notna().sum() < 10:
                return pd.Series(["NORMAL"] * len(g), index=g.index)
            p_low  = np.nanpercentile(ss, pctl_calm)
            p_high = np.nanpercentile(ss, pctl_vol)
            return pd.cut(ss, [-np.inf, p_low, p_high, np.inf], labels=["CALM","NORMAL","VOLATILE"])
        return d.groupby("symbol", group_keys=False).apply(lab, include_groups=False)

    # symbol_month (défaut)
    d["ym"] = d["t"].dt.tz_convert("UTC").dt.tz_localize(None).dt.to_period("M")
    def lab_sm(g: pd.DataFrame) -> pd.Series:
        ss = g["atr_pct"].astype(float)
        if ss.notna().sum() < 10:
            return pd.Series(["NORMAL"] * len(g), index=g.index)
        p_low  = np.nanpercentile(ss, pctl_calm)
        p_high = np.nanpercentile(ss, pctl_vol)
        return pd.cut(ss, [-np.inf, p_low, p_high, np.inf], labels=["CALM","NORMAL","VOLATILE"])
    return d.groupby(["symbol","ym"], group_keys=False).apply(lab_sm, include_groups=False)

def _get_router_params_strict(symbol: str, regime: str, table: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Strict: exige table[symbol][regime]. Aucun fallback DEFAULT."""
    sym = table.get(symbol)
    if not isinstance(sym, dict):
        return None
    p = sym.get(regime)
    return p if isinstance(p, dict) else None

def _apply_one_feature(value: float, cfg: dict) -> bool:
    if value is None or not np.isfinite(value):
        return False  # fail-closed
    t = str(cfg.get("type", "range")).lower()
    if t == "binary":
        keep = cfg.get("keep", 1)
        try:
            return int(bool(value)) == int(keep)
        except Exception:
            return False
    left  = cfg.get("left", cfg.get("min", None))
    right = cfg.get("right", cfg.get("max", None))
    ok = True
    if left  is not None: ok &= (value >= float(left))
    if right is not None: ok &= (value <= float(right))
    return bool(ok)

def _resolve_cutoffs_strict(symbol: str, regime: str, cutoffs: Dict[str, Any]) -> Optional[dict]:
    """
    Exige une entrée explicite cutoffs[symbol][regime] (dict de features).
    AUCUN fallback. Si non trouvé -> None.
    """
    sym = cutoffs.get(symbol)
    if not isinstance(sym, dict):
        return None
    cfg = sym.get(regime)
    return cfg if isinstance(cfg, dict) else None

def _build_feature_mask(
    df: pd.DataFrame,
    cutoffs: Dict[str, Any],
    return_debug: bool = False,
):
    """
    Strict: applique UNIQUEMENT cutoffs[symbol][regime_eval].
    - Fail-closed: si pas de config => tout rejeté.
    - return_debug=True => renvoie (mask, debug_row_df, debug_summary_df)
      * debug_row_df: par ligne, quelles features ont rejeté
      * debug_summary_df: agrégat par (symbol, regime, feature)
    """
    if "symbol" not in df.columns or "regime_eval" not in df.columns:
        raise ValueError("DataFrame must contain 'symbol' and 'regime_eval'.")

    mask_global = pd.Series(True, index=df.index)
    # per-row debug container (facultatif)
    row_reject_map = {}   # idx -> list[str] of feature names
    # per-feature counters
    feat_stats = []       # rows of dicts

    for (sym, reg), g in df.groupby(["symbol","regime_eval"], observed=False, sort=False):
        sym, reg = str(sym), str(reg)
        cfg = _resolve_cutoffs_strict(sym, reg, cutoffs)
        if not cfg:
            logger.warning(f"No feature cutoffs for ({sym}, {reg}) -> rejecting {len(g)} rows.")
            mask_global.loc[g.index] = False
            if return_debug:
                for idx in g.index:
                    row_reject_map[idx] = row_reject_map.get(idx, []) + ["__NO_CUTOFFS__"]
                feat_stats.append(dict(symbol=sym, regime=reg, feature="__NO_CUTOFFS__",
                                       n_rows=len(g), n_keep=0, n_reject=len(g),
                                       n_na=0, keep_rate=0.0, cfg="—"))
            continue

        # Start optimistic: keep all rows in this group
        keep_grp = pd.Series(True, index=g.index)

        for feat, fc in cfg.items():
            if not isinstance(fc, dict):
                continue
            if feat not in g.columns:
                # pas de colonne => rejet (fail-closed) pour cette feature
                keep_vec = pd.Series(False, index=g.index)
                n_na = 0
            else:
                x = pd.to_numeric(g[feat], errors="coerce")
                # applique le test élément par élément
                keep_vec = x.apply(lambda v: _apply_one_feature(v, fc))
                n_na = int(x.isna().sum())
                # si NA => _apply_one_feature renvoie False (fail-closed)

            # accumulate stats
            n_rows = int(len(g))
            n_keep = int(keep_vec.sum())
            n_rej  = int((~keep_vec).sum())
            keep_grp &= keep_vec

            if return_debug:
                # marque les lignes rejetées par CETTE feature
                rej_idx = g.index[~keep_vec]
                for idx in rej_idx:
                    row_reject_map[idx] = row_reject_map.get(idx, []) + [feat]

                # format court du cfg pour lisibilité
                t = str(fc.get("type","range"))
                left  = fc.get("left", fc.get("min"))
                right = fc.get("right", fc.get("max"))
                keep  = fc.get("keep", None)
                cfg_str = f"type={t}; min={left}; max={right}; keep={keep}"

                feat_stats.append(dict(
                    symbol=sym, regime=reg, feature=feat,
                    n_rows=n_rows, n_keep=n_keep, n_reject=n_rej, n_na=n_na,
                    keep_rate=(n_keep / n_rows if n_rows else np.nan),
                    cfg=cfg_str
                ))

        mask_global.loc[g.index] = keep_grp

    if not return_debug:
        return mask_global

    # debug_row_df: par ligne, “reject_reasons” (liste joinée)
    reject_reasons_col = pd.Series([""] * len(df), index=df.index, dtype=object)
    for idx, feats in row_reject_map.items():
        reject_reasons_col.loc[idx] = ";".join(sorted(set(feats)))

    debug_row_df = df[["t","symbol","regime_eval"]].copy()
    debug_row_df["reject_reasons"] = reject_reasons_col

    # debug_summary_df: compteurs agrégés
    debug_summary_df = pd.DataFrame(feat_stats)
    if not debug_summary_df.empty:
        debug_summary_df = debug_summary_df.sort_values(["symbol","regime","feature"])

    return mask_global, debug_row_df, debug_summary_df

# -------------- Core eval ----------------
def evaluate(
    labels_path: str,
    router_params_path: str,
    feature_cutoffs_path: str,
    out_baseline_csv: str,
    out_filtered_csv: str,
    out_summary_txt: str,
    by: str = "symbol_month",
    pctl_calm: float = 40.0,
    pctl_volatile: float = 75.0,
    min_trades_symbol: int = 30,
    storage_options: dict | None = None,
) -> None:
    logger.info("📥 Loading inputs")
    df = read_csv_any(labels_path, storage_options=storage_options)

    router_params = _open_json_s3_strict(router_params_path, storage_options=storage_options)
    feature_cutoffs = _open_json_s3_strict(feature_cutoffs_path, storage_options=storage_options)

    # Si un des deux manque: sorties vides + message, et on sort sans planter
    if router_params is None or feature_cutoffs is None:
        logger.warning("Config JSON missing -> no trading / writing empty outputs.")
        _write_empty_outputs(out_baseline_csv, out_filtered_csv, out_summary_txt, storage_options=storage_options)
        return

    # Horodatage & tri
    if "t" not in df.columns:
        raise ValueError("labels CSV must contain column 't'")
    df["t"] = pd.to_datetime(df["t"], utc=True, errors="coerce")
    df = df.dropna(subset=["t"]).sort_values("t")

    # --- Régimes (robuste & aligné, sans apply) ---
    logger.info("🏷️ Building regimes from atr_pct")
    if "atr_pct" not in df.columns:
        raise ValueError("labels CSV must contain column 'atr_pct' to build regimes.")

    df = df.drop(columns=["regime_eval"], errors="ignore")
    df["month"] = df["t"].dt.month

    # Choix des clés de regroupement
    if by == "symbol_month":
        keys = ["symbol", "month"]
    elif by == "symbol":
        keys = ["symbol"]
    else:
        keys = []  # global

    # Quantiles par groupe -> transform renvoie une série alignée au df
    if keys:
        q_lo = df.groupby(keys, observed=False)["atr_pct"].transform(
            lambda s: np.nanpercentile(s, pctl_calm) if np.isfinite(s).any() else np.nan
        )
        q_hi = df.groupby(keys, observed=False)["atr_pct"].transform(
            lambda s: np.nanpercentile(s, pctl_volatile) if np.isfinite(s).any() else np.nan
        )
    else:
        # Cas global (aucune clé)
        gl_lo = np.nanpercentile(df["atr_pct"], pctl_calm) if np.isfinite(df["atr_pct"]).any() else np.nan
        gl_hi = np.nanpercentile(df["atr_pct"], pctl_volatile) if np.isfinite(df["atr_pct"]).any() else np.nan
        q_lo = pd.Series(gl_lo, index=df.index)
        q_hi = pd.Series(gl_hi, index=df.index)

    # Fallback si un groupe est full-NaN: bascule en NORMAL
    calm_mask     = (df["atr_pct"] <= q_lo)
    volatile_mask = (df["atr_pct"] >= q_hi)
    reg = np.where(volatile_mask, "VOLATILE",
        np.where(calm_mask, "CALM", "NORMAL"))

    df["regime_eval"] = reg  # ← aligné 1:1, plus de reindex/join

    # PnL columns detection (do this BEFORE diagnostics that need pnl_cols)
    pnl_cols = _detect_all_combo_cols(df)
    if not pnl_cols:
        raise ValueError("No 'pnl_net_rr*_rf*_ts*' columns found in labels CSV.")

    # Cohérence router/combos (coverage & sanity)
    diag_router = _router_and_combo_diagnostics(df, router_params, pnl_cols)
    router_diag_path = out_summary_txt.replace(".txt","_router_combo_coverage.csv")
    write_csv_any(diag_router, router_diag_path, storage_options=storage_options)

    # Alertes rapides + garde variables pour le résumé final
    if diag_router.empty:
        missing_rp = pd.Series([], dtype=int)
        missing_combo = pd.Series([], dtype=int)
    else:
        missing_rp = diag_router["has_router_params"].eq(0)
        missing_combo = diag_router["combo_exists"].eq(0)
        if missing_rp.any():
            m = diag_router.loc[missing_rp, ["symbol","regime","n_rows"]]
            logger.warning(f"Missing router params for {len(m)} (symbol,regime) pairs. See {router_diag_path}")
        if missing_combo.any():
            m = diag_router.loc[missing_combo, ["symbol","regime","chosen_col"]]
            logger.warning(f"Router chose non-existent combo for {len(m)} pairs. See {router_diag_path}")

    # === coverage diagnostics (features présents vs attendus) ===
    pairs = []
    for (sym, reg), g in df.groupby(["symbol", "regime_eval"], observed=False):
        rp = router_params.get(str(sym), {}).get(str(reg))
        fc = feature_cutoffs.get(str(sym), {}).get(str(reg))
        feats = sorted((fc or {}).keys())
        feats_match = [f for f in feats if f in df.columns]
        pairs.append({
            "symbol": sym,
            "regime": reg,
            "n_rows": len(g),
            "has_router_params": int(isinstance(rp, dict)),
            "has_feature_cutoffs": int(isinstance(fc, dict)),
            "n_features_in_cfg": len(feats),
            "n_features_match_cols": len(feats_match),
            "missing_features": ",".join([f for f in feats if f not in df.columns])[:200],
        })
    diag = pd.DataFrame(pairs).sort_values(["symbol","regime"])
    write_csv_any(diag, out_summary_txt.replace(".txt","_diagnostics.csv"), storage_options=storage_options)

    # ---- Sélection combo par (symbol, regime) [STRICT] ----
    logger.info("🧭 Selecting combo per (symbol, regime) [strict]")
    chosen_cols: List[str] = []
    has_params: List[bool] = []
    for _, r in df.iterrows():
        p = _get_router_params_strict(str(r["symbol"]), str(r["regime_eval"]), router_params)
        if p is None:
            chosen_cols.append("")
            has_params.append(False)
        else:
            chosen_cols.append(_combo_to_col(float(p["rr_min"]), int(p["risk_floor_bps"]), int(p["time_stop"])))
            has_params.append(True)
    df["chosen_pnl_col"] = chosen_cols
    has_params = pd.Series(has_params, index=df.index)

    # Exclure lignes sans params
    n_no_params = int((~has_params).sum())
    if n_no_params:
        logger.warning(f"{n_no_params} rows have no router params (symbol/regime). Excluding them.")
    df = df[has_params].copy()

    # ---- Sélection vectorisée de la valeur PnL ----
    col_indexer = pd.Index(df.columns).get_indexer(df["chosen_pnl_col"])
    bad = (col_indexer < 0)
    if bad.any():
        n_bad = int(bad.sum())
        logger.warning(f"{n_bad} rows map to non-existent pnl columns. Excluding them.")
        df = df.loc[~bad].copy()
        col_indexer = col_indexer[~bad]

    arr = df.to_numpy()
    df["pnl_net_router"] = arr[np.arange(len(df)), col_indexer]

    # ---- Filtres micro (strict) + audit ----
    logger.info("🔎 Applying micro feature cutoffs + audit")
    mask_micro, dbg_rows, dbg_summary = _build_feature_mask(df, feature_cutoffs, return_debug=True)
    df["kept_by_micro"] = mask_micro.astype(int)

    # Sauvegardes d’audit
    base = out_summary_txt.rsplit(".txt", 1)[0]
    audit_rows_path    = base + "_audit_rows.csv"
    audit_summary_path = base + "_audit_summary.csv"
    write_csv_any(dbg_rows, audit_rows_path, storage_options=storage_options)
    write_csv_any(dbg_summary, audit_summary_path, storage_options=storage_options)
    logger.info(f"Audit rows -> {audit_rows_path}")
    logger.info(f"Audit summary -> {audit_summary_path}")

    # ---- Sauvegardes lignes (S3) ----
    base_cols = ["t","symbol","side","entry","atr_pct","regime_eval","chosen_pnl_col","pnl_net_router","kept_by_micro"]
    base_cols = [c for c in base_cols if c in df.columns]
    write_csv_any(df[base_cols], out_baseline_csv, storage_options=storage_options)
    write_csv_any(df.loc[mask_micro, base_cols], out_filtered_csv, storage_options=storage_options)

    # ---- KPIs agrégés ----
    def _agg(tbl: pd.DataFrame, label: str) -> pd.DataFrame:
        out = (
            tbl.groupby("symbol", observed=True)["pnl_net_router"]
            .agg(expectancy="mean", n="count")
            .reset_index()
            .sort_values(["expectancy"], ascending=False)
        )
        out["which"] = label
        return out

    rep_base = _agg(df, "baseline")
    rep_filt  = _agg(df.loc[mask_micro], "filtered")

    rep = rep_base.merge(rep_filt, on="symbol", how="outer", suffixes=("_base","_filt"))
    rep["uplift_E"] = rep["expectancy_filt"] - rep["expectancy_base"]
    rep["coverage"] = rep["n_filt"].fillna(0) / rep["n_base"].replace(0, np.nan)
    rep = rep.sort_values(["uplift_E"], ascending=False)

    # Guardrails symbol sans assez de trades
    few = rep["n_base"] < min_trades_symbol
    if few.any():
        logger.warning(f"{int(few.sum())} symboles avec < {min_trades_symbol} trades baseline; à vérifier.")

    # Table CSV (+ summary txt via fsspec)
    table_path = out_summary_txt.replace(".txt","_table.csv")
    write_csv_any(rep, table_path, storage_options=storage_options)

    # ---- Summary texte ----
    def _safe_mean(s):
        s = pd.to_numeric(s, errors="coerce")
        return float(np.nanmean(s)) if s.notna().any() else np.nan

    def _safe_int(x, default=0):
        try:
            if pd.isna(x):
                return default
            return int(x)
        except Exception:
            return default

    lines = []
    lines.append("==== eval_router_micro summary ====\n")
    lines.append(f"Rows total: {len(df)}\n")
    lines.append(f"Kept by micro filters: {int(mask_micro.sum())} ({mask_micro.mean():.1%})\n")
    lines.append("\n-- Global KPIs --\n")
    E_base = _safe_mean(df["pnl_net_router"])
    E_filt = _safe_mean(df.loc[mask_micro, "pnl_net_router"])
    lines.append(f"Expectancy baseline: {E_base:.6f}\n")
    lines.append(f"Expectancy filtered: {E_filt:.6f}\n")
    lines.append(f"Uplift (abs): {E_filt - E_base:+.6f}\n")
    lines.append("\n-- Per-symbol (top 10 by uplift) --\n")
    lines.append("\n-- Coverage checks --\n")
    lines.append(f"Router missing pairs: {int(missing_rp.sum())}  | Combos missing: {int(missing_combo.sum())}\n")

    for _, r in rep.head(10).iterrows():
        n_base = _safe_int(r.get("n_base"), 0)
        n_filt = _safe_int(r.get("n_filt"), 0)
        exp_base = r.get("expectancy_base", np.nan)
        exp_filt = r.get("expectancy_filt", np.nan)
        uplift = r.get("uplift_E", np.nan)
        cov = r.get("coverage", np.nan)

        lines.append(
            f"{r['symbol']}: "
            f"E_base={float(exp_base):.6f} (n={n_base}) "
            f"→ E_filt={float(exp_filt):.6f} (n={n_filt}) "
            f"| uplift={float(uplift):+.6f} | coverage={0.0 if pd.isna(cov) else float(cov):.2%}\n"
        )

    write_text_any("".join(lines), out_summary_txt, storage_options=storage_options)

    logger.info("✅ Done.")
    logger.info(f"Baseline rows -> {out_baseline_csv}")
    logger.info(f"Filtered rows -> {out_filtered_csv}")
    logger.info(f"Summary -> {out_summary_txt}")
    logger.info(f"Per-symbol table -> {table_path}")

# -------------- CLI ----------------------
def parse_args():
    p = argparse.ArgumentParser(description="Evaluate router performance using multigrid+micro CSV (no L1 replay).")
    p.add_argument("--labels", required=True)
    p.add_argument("--router-json", required=True)
    p.add_argument("--feature-cutoffs", required=True)
    p.add_argument("--out-baseline", required=True)
    p.add_argument("--out-filtered", required=True)
    p.add_argument("--out-summary", required=True)
    p.add_argument("--by", choices=["global","symbol","symbol_month"], default="symbol_month")
    p.add_argument("--pctl-calm", type=float, default=40.0)
    p.add_argument("--pctl-volatile", type=float, default=75.0)
    p.add_argument("--min-trades-symbol", type=int, default=30)
    # S3 auth (optional)
    p.add_argument("--s3-anon", action="store_true")
    p.add_argument("--s3-profile", type=str, default=None)
    p.add_argument("--s3-region", type=str, default=None)
    return p.parse_args()

def main():
    args = parse_args()
    storage_options: Dict[str, Any] = {"anon": bool(args.s3_anon)}
    if args.s3_profile:
        storage_options["profile"] = args.s3_profile
    if args.s3_region:
        storage_options["client_kwargs"] = {"region_name": args.s3_region}

    evaluate(
        labels_path=args.labels,
        router_params_path=args.router_json,
        feature_cutoffs_path=args.feature_cutoffs,
        out_baseline_csv=args.out_baseline,
        out_filtered_csv=args.out_filtered,
        out_summary_txt=args.out_summary,
        by=args.by,
        pctl_calm=args.pctl_calm,
        pctl_volatile=args.pctl_volatile,
        min_trades_symbol=args.min_trades_symbol,
        storage_options=storage_options,
    )

if __name__ == "__main__":
    main()