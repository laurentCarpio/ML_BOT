#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
validate_stage2.py — Audit/validation des fichiers Stage2 (micro + labels) sur S3, memory-safe.

NOUVELLE ARBO (issue de build_stage2_dataset.py) :
  s3://.../data/stage2/{train|val|test}/<SYMBOL>/<YEAR>/parts/part-*.parquet

Exemples :
  # 1) Auditer les trois splits (train/val/test) pour tous symboles/années
  python validate_stage2.py \
    --src-root s3://tradebot-config-tokyo/data/stage2 \
    --split all \
    --s3-region ap-northeast-1

  # 2) Auditer seulement TRAIN pour un sous-ensemble (BTCUSDT,APTUSDT ; 2023 & 2024)
  python validate_stage2.py \
    --src-root s3://tradebot-config-tokyo/data/stage2 \
    --split train \
    --symbols BTCUSDT APTUSDT \
    --years 2023 2024 \
    --s3-region ap-northeast-1
"""

from __future__ import annotations
import argparse, re, math, sys, gc
from collections import Counter, defaultdict
from typing import Iterable, List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import fsspec
from botocore.exceptions import ClientError

# ----------------------------
# Références/colonnes
# ----------------------------
TF_HORIZONS_REF = {
    "5m":  360,
    "15m": 1080,
    "30m": 2160,
    "1h":  4320,
    "2h":  7200,
    "4h":  14400,
}

# colonnes numériques attendues (pour min/max/NaN rate)
# colonnes numériques attendues (pour min/max/NaN rate)
NUM_LIKE = [
    "spread_bps_entry","quote_churn_10s",
    "cum_depth_within_5bps_opp","cum_depth_within_10bps_opp",
    "wall_opp_share_5","wall_opp_share_15",
    "obi_5","obi_15","microprice_bias",
    "slope_bid_5","slope_ask_5","slope_bid_15","slope_ask_15",
    "aggr_ratio_10s","aggr_ratio_15s","net_delta_15s",
    "ret_stdev_1s_10s_bps","mid_jump_bps_3s",
    "bb_width","adx","atr_percentile","fees_bps","atr_bps","exchange_latency_ms",
    "imbalance_persistence","executed_vs_added_ratio",
    "horizon_sec","pnl_net_max_bps","pnl_net_min_bps","THRESH_BPS",
    "obi_5_side","obi_15_side","microprice_bias_side",
    "slope_bid_5_side","slope_ask_5_side",
    "slope_bid_15_side","slope_ask_15_side",
    "aggr_ratio_10s_side","aggr_ratio_15s_side",
    "net_delta_15s_side","mid_jump_bps_3s_side",

    # 🔥 NOUVELLES FEATURES MICROSTRUCTURE SHORT/LONG
    # Imbalance d'agression
    "imbl_aggr_3s",
    "imbl_aggr_10s",
    "imbl_aggr_3s_side",
    "imbl_aggr_10s_side",

    # Microprice bias lissé
    "microprice_bias_ma_1s",
    "microprice_bias_ma_3s",
    "microprice_bias_ma_1s_side",
    "microprice_bias_ma_3s_side",

    # Spread & vacuum
    "delta_spread_3s",
    "best_bid_refill_rate",

    # VWAP / pullback
    "mid_minus_vwap_3s",
    "mid_minus_vwap_3s_side",

    # Volatilité / régime
    "atr_pct_rank_30m",
    "bb_width_pctl",

    # Walls & decay
    "ask_wall_decay_3s",

    # LONG-spécifiques (absorption / refill)
    "bid_absorb_ratio_3s",
    "bid_absorb_ratio_10s",
    "bid_refill_persistence_ticks",
]

CAT_LIKE = ["symbol","tf","side"]
INT_LIKE = ["Y","year","spread_widen_flag_10s"]  # Y∈{-1,0,1}

# ----------------------------
# Logs/IO
# ----------------------------
def _log(msg: str): print(msg, flush=True)

def _so(region: Optional[str], anon: bool, requester_pays: bool) -> dict:
    so: Dict[str, object] = {}
    if region: so["client_kwargs"] = {"region_name": region}
    if anon:   so["anon"] = True
    if requester_pays: so["requester_pays"] = True
    return so

def _open_parquet_file(path: str, so: dict) -> pq.ParquetFile:
    fo = fsspec.open(path, "rb", **(so or {})).open()
    return pq.ParquetFile(fo)

# ----------------------------
# Découverte des fichiers S3
# ----------------------------
def _glob_split_paths(src_root: str,
                      split: str,
                      so: dict,
                      symbols: Optional[List[str]],
                      years: Optional[List[int]],
                      debug: bool=False) -> List[str]:
    """
    Uniquement ce motif (exigé) :
      {src_root}/{split}/{SYM}/{YEAR}/parts/part-*.parquet
    """
    src_root = src_root.rstrip("/")
    syms = symbols or ["*"]
    yrs  = [str(y) for y in (years or ["*"])]

    patterns: List[str] = []
    for s in syms:
        for y in yrs:
            patterns.append(f"{src_root}/{split}/{s}/{y}/parts/part-*.parquet")

    all_paths: List[str] = []
    for pat in patterns:
        if debug:
            _log(f"[glob] try: {pat}")
        try:
            fs, _, paths = fsspec.get_fs_token_paths(pat, storage_options=so)
            proto = fs.protocol[0] if isinstance(fs.protocol, (list, tuple)) else fs.protocol
            matches = []
            for p in sorted(paths):
                if not re.match(r"^[a-z0-9]+://", str(p)):
                    p = f"{proto}://{p}"
                matches.append(p)
            if debug:
                _log(f"[glob]  -> {len(matches)} match(es)")
            all_paths.extend(matches)
        except PermissionError:
            _log(f"❌ AccessDenied lors du glob: {pat} — vérifie s3:ListBucket, ou passe des chemins exacts.")
        except ClientError as e:
            _log(f"❌ ClientError lors du glob: {pat} — {e}")

    uniq = sorted(set(all_paths))
    if debug:
        _log(f"[glob] total unique matches for split='{split}': {len(uniq)}")
    return uniq

# ----------------------------
# Contrôles (inchangés de ton script)
# ----------------------------
def assert_schema_consistency(paths: list[str], so: dict) -> None:
    if not paths: return
    with fsspec.open(paths[0], "rb", **(so or {})) as f:
        df0 = pd.read_parquet(f)
    ref_cols = list(df0.columns)
    ref_dtypes = df0.dtypes.astype(str).to_dict()

    bad = []
    for p in paths[1:]:
        try:
            with fsspec.open(p, "rb", **(so or {})) as f:
                dfi = pd.read_parquet(f)
        except Exception as e:
            bad.append((p, f"read_error={e!r}")); continue
        cols = list(dfi.columns)
        dts = dfi.dtypes.astype(str).to_dict()
        missing = [c for c in ref_cols if c not in cols]
        extra   = [c for c in cols if c not in ref_cols]
        dtype_mismatch = {c:(ref_dtypes[c], dts[c]) for c in ref_cols if c in dts and ref_dtypes[c] != dts[c]}
        if missing or extra or dtype_mismatch:
            bad.append((p, f"missing={missing}, extra={extra}, dtype_mismatch={dtype_mismatch}"))

    if bad:
        _log("❌ Schéma/dtypes inconsistents:")
        for p, why in bad:
            _log(f"- {p}\n  ↳ {why}")
    else:
        _log("✅ Schéma/dtypes homogènes sur toutes les parts.")

def _fmt_ratio(ok: int, tot: int) -> str:
    return f"{ok}/{tot} = {ok/tot:.2%}" if tot else "0/0"

class Metrics:
    def __init__(self):
        self.rows = 0
        self.nulls = Counter()
        self.numeric_min = defaultdict(lambda: math.inf)
        self.numeric_max = defaultdict(lambda: -math.inf)
        self.tf_counts = Counter()
        self.side_counts = Counter()
        self.y_counts = Counter()
        self.year_counts = Counter()
        self.symbol_counts = Counter()

        self.chk_label_ok = 0
        self.chk_label_total = 0
        self.chk_horizon_ok = 0
        self.chk_horizon_total = 0
        self.bad_tfs = Counter()

    def update_batch(self, df: pd.DataFrame):
        if df.empty: return
        n = len(df)
        self.rows += n

        for c in ["tf","side","symbol","year","Y"]:
            if c in df.columns:
                vc = df[c].value_counts(dropna=False)
                if c == "tf":     self.tf_counts.update(vc.to_dict())
                elif c == "side": self.side_counts.update(vc.to_dict())
                elif c == "symbol": self.symbol_counts.update(vc.to_dict())
                elif c == "year":   self.year_counts.update(vc.to_dict())
                elif c == "Y":      self.y_counts.update(vc.to_dict())

        null_series = df.isna().sum()
        for c, k in null_series.to_dict().items():
            self.nulls[c] += int(k)

        for c in NUM_LIKE:
            if c in df.columns:
                col = pd.to_numeric(df[c], errors="coerce")
                col_min = np.nanmin(col.values) if col.notna().any() else math.inf
                col_max = np.nanmax(col.values) if col.notna().any() else -math.inf
                if col_min < self.numeric_min[c]: self.numeric_min[c] = float(col_min)
                if col_max > self.numeric_max[c]: self.numeric_max[c] = float(col_max)

        need = {"Y","THRESH_BPS","pnl_net_max_bps","pnl_net_min_bps"}
        if need.issubset(df.columns):
            y = pd.to_numeric(df["Y"], errors="coerce")
            th = pd.to_numeric(df["THRESH_BPS"], errors="coerce")
            mx = pd.to_numeric(df["pnl_net_max_bps"], errors="coerce")
            mn = pd.to_numeric(df["pnl_net_min_bps"], errors="coerce")
            mask = y.notna() & th.notna() & mx.notna() & mn.notna()
            self.chk_label_total += int(mask.sum())
            if mask.any():
                yv, thv, mxv, mnv = y[mask], th[mask], mx[mask], mn[mask]
                ok = (
                    ((yv == 1)  & (mxv >  thv)) |
                    ((yv == -1) & (mnv < -thv)) |
                    ((yv == 0)  & ~( (mxv > thv) | (mnv < -thv) ))
                ).sum()
                self.chk_label_ok += int(ok)

        if "tf" in df.columns and "horizon_sec" in df.columns:
            tf = df["tf"].astype(str)
            hz = pd.to_numeric(df["horizon_sec"], errors="coerce")
            mask = tf.notna() & hz.notna()
            self.chk_horizon_total += int(mask.sum())
            if mask.any():
                for tfi, hzi in zip(tf[mask].values, hz[mask].values):
                    if tfi in TF_HORIZONS_REF:
                        if int(hzi) == TF_HORIZONS_REF[tfi]:
                            self.chk_horizon_ok += 1
                    else:
                        self.bad_tfs[tfi] += 1

class StreamingChecks:
    def __init__(self, nulls_threshold_pct: float = 5.0, var_eps: float = 1e-12, max_keys:int=5_000_000):
        self.key_seen = set(); self.max_keys = max_keys; self.dupes_count = 0
        self.h_ok = defaultdict(int); self.h_tot = defaultdict(int)
        self.nulls_counts = defaultdict(lambda: defaultdict(int))
        self.group_sizes = defaultdict(int)
        self.var_eps = var_eps; self.nulls_threshold_pct = nulls_threshold_pct
        self.w_count = defaultdict(lambda: defaultdict(int))
        self.w_mean  = defaultdict(lambda: defaultdict(float))
        self.w_M2    = defaultdict(lambda: defaultdict(float))

    def process_batch(self, df: pd.DataFrame):
        if df is None or df.empty: return

        key_cols = ("symbol","year","t","tf","side")
        if all(c in df.columns for c in key_cols):
            for tpl in zip(*(df[c].astype(str).values for c in key_cols)):
                h = hash(tpl)
                if h in self.key_seen: self.dupes_count += 1
                elif len(self.key_seen) < self.max_keys: self.key_seen.add(h)

        if {"tf","horizon_sec"}.issubset(df.columns):
            h = df["horizon_sec"].astype("Int64")
            tf = df["tf"].astype(str)
            for tfi, hi in zip(tf.values, h.values):
                self.h_tot[tfi] += 1
                ref = TF_HORIZONS_REF.get(tfi)
                if ref is not None and pd.notna(hi) and int(hi) == ref:
                    self.h_ok[tfi] += 1

        if {"symbol","tf"}.issubset(df.columns):
            for (sym, tf), g in df.groupby(["symbol","tf"], sort=False):
                n = len(g); self.group_sizes[(sym, tf)] += n
                na_counts = g.isna().sum()
                for col, c in na_counts.items():
                    self.nulls_counts[(sym, tf)][col] += int(c)

        if "symbol" in df.columns:
            num_cols = df.select_dtypes(include=[np.number]).columns
            if len(num_cols):
                for sym, g in df.groupby("symbol", sort=False):
                    for col in num_cols:
                        x = g[col].to_numpy(dtype=float, copy=False)
                        m = np.isfinite(x); x = x[m]
                        for xi in x:
                            cnt = self.w_count[sym][col] + 1
                            mean = self.w_mean[sym][col]
                            delta = xi - mean
                            mean += delta / cnt
                            delta2 = xi - mean
                            M2 = self.w_M2[sym][col] + delta * delta2
                            self.w_count[sym][col] = cnt
                            self.w_mean[sym][col]  = mean
                            self.w_M2[sym][col]    = M2

    def report(self):
        if self.dupes_count > 0: _log(f"❌ Doublons (symbol,year,t,tf,side): {self.dupes_count}")
        else: _log("✅ Aucun doublon (symbol,year,t,tf,side).")

        _log("-- Check horizon_sec par TF (référence validator)")
        all_tf = sorted(set(list(self.h_tot.keys()) + list(TF_HORIZONS_REF.keys())))
        for tf in all_tf:
            tot = self.h_tot.get(tf, 0)
            ok  = self.h_ok.get(tf, 0)
            pct = (100.0*ok/max(tot,1)) if tot else float("nan")
            ref = TF_HORIZONS_REF.get(tf)
            _log(f"{tf}: {('nan' if not np.isfinite(pct) else f'{pct:.2f}%')} ({ok}/{tot}) ref={ref if ref is not None else 'NA'}")

        # Nulls par (symbol,tf) au-dessus d’un seuil
        any_flag = False
        for key, tot in self.group_sizes.items():
            sym, tf = key; na_map = self.nulls_counts[key]
            bad = [(col, (cnt*100.0/max(tot,1))) for col, cnt in na_map.items()
                   if (cnt*100.0/max(tot,1)) > self.nulls_threshold_pct]
            bad.sort(key=lambda kv: kv[1], reverse=True)
            if bad:
                any_flag = True
                _log(f"⚠️ Nulls > {self.nulls_threshold_pct:.1f}% pour {sym}/{tf} — top 10 :")
                for col, pct in bad[:10]:
                    _log(f"  {col}: {pct:.2f}%")
                if len(bad) > 10: _log(f"  ... (+{len(bad)-10} colonnes)")
        if not any_flag:
            _log(f"✅ Nulls par (symbol,tf): aucune colonne > {self.nulls_threshold_pct:.1f}%")

        # Variance quasi nulle
        for sym, colmap in self.w_count.items():
            dead = []
            for col, cnt in colmap.items():
                if cnt < 2: continue
                M2 = self.w_M2[sym][col]
                var = M2 / cnt
                if var < 1e-12: dead.append(col)
            if dead:
                _log(f"⚠️ {sym}: {len(dead)} colonnes quasi constantes (var<1e-12): " +
                     ", ".join(dead[:20]) + (" ..." if len(dead) > 20 else ""))
            else:
                _log(f"✅ {sym}: pas de colonne quasi constante (var<1e-12).")

# ----------------------------
# Validation d’un split
# ----------------------------
def _validate_paths(paths: List[str], so: dict, batch_rows: int) -> None:
    if not paths:
        _log("⚠️ Aucun fichier à valider pour ce split."); return

    _log(f"→ {len(paths)} fichier(s) à valider :")
    for p in paths[:20]:
        _log(f"   - {p}")
    if len(paths) > 20:
        _log(f"   ... (+{len(paths)-20} autres)")

    # schéma homogène ?
    assert_schema_consistency(paths, so)

    metrics = Metrics()
    schk = StreamingChecks(nulls_threshold_pct=5.0, var_eps=1e-12, max_keys=5_000_000)

    for p in paths:
        pf = _open_parquet_file(p, so)
        cols = list(pf.schema.names)
        for batch in pf.iter_batches(batch_size=batch_rows, columns=cols):
            t = pa.Table.from_batches([batch])
            df = t.to_pandas(types_mapper=pd.ArrowDtype)

            metrics.update_batch(df)
            schk.process_batch(df)

            del df, t, batch
            gc.collect()

    # Reporting
    print("\n===== RÉSUMÉ (split) =====")
    print(f"Total rows: {metrics.rows:,}")

    print("\n-- Répartition Y --")
    for k,v in metrics.y_counts.most_common():
        print(f"Y={k}: {v:,}")

    print("\n-- Répartition TF --")
    for k,v in metrics.tf_counts.most_common():
        print(f"{k}: {v:,}")

    print("\n-- Répartition side --")
    for k,v in metrics.side_counts.most_common():
        print(f"{k}: {v:,}")

    print("\n-- Répartition symbol --")
    for k,v in metrics.symbol_counts.most_common():
        print(f"{k}: {v:,}")

    print("\n-- Années --")
    for k,v in sorted(metrics.year_counts.items()):
        print(f"{k}: {v:,}")

    print("\n-- Taux de nulls par colonne (top 30) --")
    for c,n in metrics.nulls.most_common(30):
        rate = n / metrics.rows if metrics.rows else 0.0
        print(f"{c:30s} {n:>10,d}  ({rate:.2%})")

    print("\n-- Bornes numériques (min .. max) --")
    for c in NUM_LIKE:
        mn = metrics.numeric_min.get(c, math.inf)
        mx = metrics.numeric_max.get(c, -math.inf)
        if mn is not math.inf and mx is not -math.inf:
            print(f"{c:30s} {mn:>12.6g} .. {mx:>12.6g}")

    print("\n-- Check label (Y vs THRESH_BPS/pnl path) --")
    print(_fmt_ratio(metrics.chk_label_ok, metrics.chk_label_total))

    print("\n-- Check horizon_sec par TF --")
    print(_fmt_ratio(metrics.chk_horizon_ok, metrics.chk_horizon_total))
    if metrics.bad_tfs:
        print("TF non mappés (à inspecter):", dict(metrics.bad_tfs))

    # Sanity rapides
    alarms = []
    tot_y = sum(v for k,v in metrics.y_counts.items() if pd.notna(k))
    maj_prop = max((v / tot_y) for k,v in metrics.y_counts.items()) if tot_y else 0
    if maj_prop > 0.9:
        alarms.append(f"Distribution Y très déséquilibrée (classe majoritaire ≈ {maj_prop:.1%}).")
    if ("microprice_bias" in metrics.numeric_min and
        (metrics.numeric_min["microprice_bias"] < -0.1 or metrics.numeric_max["microprice_bias"] > 0.1)):
        alarms.append("microprice_bias extrêmes au-delà de ±10% (vérifier unités/calcul).")
    if ("fees_bps" in metrics.numeric_min and
        (metrics.numeric_min["fees_bps"] < 0 or metrics.numeric_max["fees_bps"] > 50)):
        alarms.append("fees_bps hors [0,50] bps ?")
    if "THRESH_BPS" in metrics.numeric_min and metrics.numeric_min["THRESH_BPS"] <= 0:
        alarms.append("Certain(s) THRESH_BPS ≤ 0 (devrait être > 0)")
    
    # --- Sanity spécifiques nouvelles features ---
    # Imbalance d'agression [0,1] + version side [-1,1]
    for col in ("imbl_aggr_3s", "imbl_aggr_10s"):
        if col in metrics.numeric_min:
            mn = metrics.numeric_min[col]
            mx = metrics.numeric_max[col]
            if mn < -0.05 or mx > 1.05:  # petite marge pour le bruit
                alarms.append(f"{col}: valeurs hors [0,1] (min={mn:.3g}, max={mx:.3g}) — vérifier calcul / clip.")

    for col in ("imbl_aggr_3s_side", "imbl_aggr_10s_side",
                "aggr_ratio_10s_side", "aggr_ratio_15s_side",
                "net_delta_15s_side",
                "microprice_bias_ma_1s_side", "microprice_bias_ma_3s_side",
                "mid_minus_vwap_3s_side"):
        if col in metrics.numeric_min:
            mn = metrics.numeric_min[col]
            mx = metrics.numeric_max[col]
            if mn < -1.2 or mx > 1.2:
                alarms.append(f"{col}: valeurs hors [-1,1] (min={mn:.3g}, max={mx:.3g}) — vérifier mapping *side*.")

    # microprice_bias_ma_* devrait rester petit (qq dizaines de bps max)
    for col in ("microprice_bias_ma_1s", "microprice_bias_ma_3s"):
        if col in metrics.numeric_min:
            mn = metrics.numeric_min[col]
            mx = metrics.numeric_max[col]
            if mn < -0.05 or mx > 0.05:
                alarms.append(f"{col}: amplitude > 5% (min={mn:.3g}, max={mx:.3g}) — unités OK ?")

    # mid_minus_vwap_3s en bps : éviter les délires > 1000 bps
    if "mid_minus_vwap_3s" in metrics.numeric_min:
        mn = metrics.numeric_min["mid_minus_vwap_3s"]
        mx = metrics.numeric_max["mid_minus_vwap_3s"]
        if mn < -1000 or mx > 1000:
            alarms.append(f"mid_minus_vwap_3s: amplitude > 1000 bps (min={mn:.3g}, max={mx:.3g}) — vérifier normalisation/échelle.")

    # Ratios absorption/refill: >=0 et pas astronomiques
    for col in ("bid_absorb_ratio_3s", "bid_absorb_ratio_10s",
                "executed_vs_added_ratio", "best_bid_refill_rate"):
        if col in metrics.numeric_min:
            mn = metrics.numeric_min[col]
            mx = metrics.numeric_max[col]
            if mn < -1e-6 or mx > 100:
                alarms.append(f"{col}: valeurs hors [0,100] (min={mn:.3g}, max={mx:.3g}) — clip recommandé.")

    # bb_width_pctl / atr_pct_rank_30m ∈ [0,1]
    for col in ("bb_width_pctl", "atr_pct_rank_30m"):
        if col in metrics.numeric_min:
            mn = metrics.numeric_min[col]
            mx = metrics.numeric_max[col]
            if mn < -0.01 or mx > 1.01:
                alarms.append(f"{col}: devrait être un percentile ∈ [0,1], min={mn:.3g}, max={mx:.3g}.")

    if alarms:
        print("\n-- ALARMES / REMARQUES --")
        for a in alarms: print("•", a)

    print("\n-- Contrôles avancés (streaming) --")
    schk.report()
    print("\n✅ Validation (split) terminée.\n")

# ----------------------------
# Orchestration multi-splits
# ----------------------------
# --- remplace run_validate(...) par : ---
def run_validate(src_root: str,
                 split: str,
                 symbols: Optional[List[str]],
                 years: Optional[List[int]],
                 s3_region: Optional[str],
                 s3_anon: bool,
                 s3_requester_pays: bool,
                 batch_rows: int,
                 debug: bool):
    so = _so(s3_region, s3_anon, s3_requester_pays)

    splits = ["train","val","test"] if split == "all" else [split]
    for sp in splits:
        print("="*80)
        print(f"🔎 Split: {sp}")
        print("="*80)
        paths = _glob_split_paths(src_root, sp, so, symbols, years, debug=debug)
        if not paths:
            _log(f"⚠️ Aucun fichier trouvé pour split={sp}")
        else:
            _log(f"✅ {len(paths)} fichier(s) trouvé(s) pour split={sp}")
        _validate_paths(paths, so, batch_rows)

# ----------------------------
# CLI
# ----------------------------
# --- remplace parse_args() par : ---
def parse_args():
    p = argparse.ArgumentParser("Validate Stage2 parquet(s) on S3 (train/val/test)")
    p.add_argument("--src-root", required=True, help="s3://.../data/stage2")
    p.add_argument("--split", choices=["all","train","val","test"], default="all", help="Splits à valider (all par défaut)")
    p.add_argument("--symbols", nargs="*", default=None, help="Ex: BTCUSDT ETHUSDT APTUSDT")
    p.add_argument("--years", nargs="*", type=int, default=None, help="Ex: 2023 2024 2025")
    p.add_argument("--s3-region", default="ap-northeast-1")
    p.add_argument("--s3-anon", action="store_true")
    p.add_argument("--s3-requester-pays", action="store_true")
    p.add_argument("--batch-rows", type=int, default=100_000,
                   help="Taille batch Arrow (équilibre mémoire/performances)")
    p.add_argument("--debug", action="store_true")
    return p.parse_args()

# --- et dans le __main__ ---
if __name__ == "__main__":
    args = parse_args()
    run_validate(args.src_root, args.split, args.symbols, args.years,
                 args.s3_region, args.s3_anon, args.s3_requester_pays, args.batch_rows,
                 args.debug)