#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
validate_stage2.py — Validation Stage2 v3.1 (label_type go/dir, side-fixed buy/sell), memory-safe.

Arbo supportée :
  s3://.../data/stage2/{train|val|test|all}/<SYMBOL>/<YEAR>/parts/part-*.parquet

Checks clés (Stage2 v3.1):
  - colonnes requises
  - distributions Y / tf / side / label_type / task
  - cohérence label_type=go: paires buy+sell par (symbol,year,t,tf,label_type) + mapping (y_raw -> Y par side)
  - cohérence label_type=dir: y_raw != 0 par défaut (optionnel), + mapping y_raw->Y row-level
  - cohérence task = f"{label_type}_{tf}" (sauf legacy)
  - bornes numériques (min/max) + taux de NaN
  - doublons sur (symbol,year,t,tf,label_type,side)
"""

from __future__ import annotations

import argparse
import re
import math
import gc
from collections import Counter, defaultdict
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import fsspec
from botocore.exceptions import ClientError


# ----------------------------
# Colonnes attendues Stage2 v3.1
# ----------------------------
REQUIRED_COLS = [
    "t", "symbol", "year", "tf",
    "label_type", "task", "y_raw",
    "side", "side_num",
    "Y",
    "entry", "bid_entry", "ask_entry", "spread_bps_entry", "atr_bps",
    "mid_t", "spread_bps_top",
]

# Numeric columns (min/max/NaN rate)
NUM_LIKE = [
    "entry","bid_entry","ask_entry","spread_bps_entry","atr_bps",
    "mid_t","spread_bps_top",
    "obi_5","obi_15","microprice_bias",
    "slope_bid_5","slope_ask_5","slope_bid_15","slope_ask_15",
    "wall_opp_share_5","wall_opp_share_15",
    "cum_depth_within_5bps_opp","cum_depth_within_10bps_opp",
    "quote_churn_10s","ret_stdev_1s_10s_bps",
    "aggr_ratio_10s","aggr_ratio_15s",
    "bt_dom_3s","bt_dom_5s","bt_dom_10s",
    "obi_5_side","obi_15_side","microprice_bias_side",
    "aggr_ratio_10s_side","aggr_ratio_15s_side",
    "bt_dom_3s_side","bt_dom_10s_side",

    # audit-only (souvent NaN côté perdant)
    "audit_tp_bps","audit_sl_bps","audit_rr_min","audit_risk_r_bps","audit_pnl_net_bps",
]

CAT_LIKE = ["symbol", "tf", "side", "label_type", "task", "audit_exit_reason"]
INT_LIKE = ["Y", "y_raw", "year", "side_num"]


# ----------------------------
# Logs / IO
# ----------------------------
def _log(msg: str):
    print(msg, flush=True)

def _so(region: Optional[str], anon: bool, requester_pays: bool) -> dict:
    so: Dict[str, object] = {}
    if region:
        so["client_kwargs"] = {"region_name": region}
    if anon:
        so["anon"] = True
    if requester_pays:
        so["requester_pays"] = True
    return so

def _open_parquet_file(path: str, so: dict) -> pq.ParquetFile:
    fo = fsspec.open(path, "rb", **(so or {})).open()
    return pq.ParquetFile(fo)


# ----------------------------
# Découverte des fichiers S3
# ----------------------------
def _glob_split_paths(
    src_root: str,
    split: str,
    so: dict,
    symbols: Optional[List[str]],
    years: Optional[List[int]],
    debug: bool = False
) -> List[str]:
    src_root = src_root.rstrip("/")
    syms = symbols or ["*"]
    yrs  = [str(y) for y in (years or ["*"])]

    patterns: List[str] = [
        f"{src_root}/{split}/{s}/{y}/parts/part-*.parquet"
        for s in syms for y in yrs
    ]

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
            _log(f"❌ AccessDenied glob: {pat} — vérifie s3:ListBucket ou passe chemins exacts.")
        except ClientError as e:
            _log(f"❌ ClientError glob: {pat} — {e}")

    uniq = sorted(set(all_paths))
    if debug:
        _log(f"[glob] total unique matches split='{split}': {len(uniq)}")
    return uniq


# ----------------------------
# Checks: schéma homogène (sans lire tout en pandas)
# ----------------------------
def assert_schema_consistency(paths: List[str], so: dict) -> None:
    if not paths:
        return

    pf0 = _open_parquet_file(paths[0], so)

    # Arrow schema = stable API (names + types)
    s0 = pf0.schema_arrow
    ref_cols = list(s0.names)
    ref_types = {name: str(s0.field(name).type) for name in ref_cols}

    bad = []
    for p in paths[1:]:
        try:
            pfi = _open_parquet_file(p, so)
            si = pfi.schema_arrow
            cols = list(si.names)
            types = {name: str(si.field(name).type) for name in cols}
        except Exception as e:
            bad.append((p, f"read_error={e!r}"))
            continue

        missing = [c for c in ref_cols if c not in cols]
        extra   = [c for c in cols if c not in ref_cols]

        dtype_mismatch = {}
        for c in ref_cols:
            if c in types and ref_types.get(c) != types.get(c):
                dtype_mismatch[c] = (ref_types[c], types[c])

        if missing or extra or dtype_mismatch:
            bad.append((p, f"missing={missing}, extra={extra}, dtype_mismatch={dtype_mismatch}"))

    if bad:
        _log("❌ Schéma/types Arrow inconsistents:")
        for p, why in bad[:20]:
            _log(f"- {p}\n  ↳ {why}")
        if len(bad) > 20:
            _log(f"... (+{len(bad)-20} autres mismatches)")
    else:
        _log("✅ Schéma/types Arrow homogènes sur toutes les parts.")

def _fmt_ratio(ok: int, tot: int) -> str:
    return f"{ok}/{tot} = {ok/tot:.2%}" if tot else "0/0"

# ----------------------------
# Metrics streaming
# ----------------------------
class Metrics:
    def __init__(self):
        self.rows = 0
        self.nulls = Counter()
        self.numeric_min = defaultdict(lambda: math.inf)
        self.numeric_max = defaultdict(lambda: -math.inf)

        self.tf_counts = Counter()
        self.side_counts = Counter()
        self.y_counts = Counter()
        self.yraw_counts = Counter()
        self.label_type_counts = Counter()
        self.task_counts = Counter()
        self.year_counts = Counter()
        self.symbol_counts = Counter()

        self.tf_by_label = Counter()  # (label_type, tf) -> count
        self.y_by_label = Counter()   # (label_type, Y) -> count

        self.chk_required_ok = 0
        self.chk_required_total = 0

        # label logic checks
        self.chk_go_pair_total = 0
        self.chk_go_pair_ok = 0

        self.chk_go_mapping_total = 0
        self.chk_go_mapping_ok = 0

        self.chk_dir_neutral_total = 0
        self.chk_dir_neutral_ok = 0

        self.chk_dir_mapping_total = 0
        self.chk_dir_mapping_ok = 0

        self.chk_task_total = 0
        self.chk_task_ok = 0

        self.chk_ranges_ok = 0
        self.chk_ranges_total = 0

        # for pair-building (approx streaming)
        self._pair_cache = {}  # key -> dict(side -> (Y, y_raw))
        self._pair_cache_max = 2_000_000

    def _cache_pair(self, key, side, y, yraw):
        d = self._pair_cache.get(key)
        if d is None:
            if len(self._pair_cache) >= self._pair_cache_max:
                return
            d = {}
            self._pair_cache[key] = d
        d[side] = (y, yraw)

    def _flush_pairs(self):
        for key, d in self._pair_cache.items():
            lt = key[-1]
            if lt != "go":
                continue
            self.chk_go_pair_total += 1
            ok_pair = ("buy" in d) and ("sell" in d)
            if ok_pair:
                self.chk_go_pair_ok += 1
                # mapping check
                yb, yraw_b = d["buy"]
                ys, yraw_s = d["sell"]
                self.chk_go_mapping_total += 1
                ok = True
                if yraw_b != yraw_s:
                    ok = False
                else:
                    yr = int(yraw_b)
                    if yr == 0:
                        ok = (yb == 0 and ys == 0)
                    elif yr == 1:
                        ok = (yb == 1 and ys == 0)
                    elif yr == -1:
                        ok = (yb == 0 and ys == 1)
                    else:
                        ok = False
                if ok:
                    self.chk_go_mapping_ok += 1
        self._pair_cache.clear()

    def update_batch(self, df: pd.DataFrame, dir_include_neutral: bool):
        if df is None or df.empty:
            return
        n = len(df)
        self.rows += n

        # required cols check
        self.chk_required_total += n
        if all(c in df.columns for c in REQUIRED_COLS):
            ok = df[REQUIRED_COLS].notna().all(axis=1).sum()
            self.chk_required_ok += int(ok)

        # basic counts
        for c in ["tf", "side", "symbol", "year", "Y", "y_raw", "label_type", "task"]:
            if c in df.columns:
                vc = df[c].value_counts(dropna=False)
                if c == "tf":
                    self.tf_counts.update(vc.to_dict())
                elif c == "side":
                    self.side_counts.update(vc.to_dict())
                elif c == "symbol":
                    self.symbol_counts.update(vc.to_dict())
                elif c == "year":
                    self.year_counts.update(vc.to_dict())
                elif c == "Y":
                    self.y_counts.update(vc.to_dict())
                elif c == "y_raw":
                    self.yraw_counts.update(vc.to_dict())
                elif c == "label_type":
                    self.label_type_counts.update(vc.to_dict())
                elif c == "task":
                    self.task_counts.update(vc.to_dict())

        # by label_type summaries
        if {"label_type", "tf"}.issubset(df.columns):
            tmp = df[["label_type","tf"]].astype(str)
            for lt, tf in zip(tmp["label_type"].values, tmp["tf"].values):
                self.tf_by_label[(lt, tf)] += 1

        if {"label_type", "Y"}.issubset(df.columns):
            tmp = df[["label_type","Y"]]
            for lt, y in zip(tmp["label_type"].astype(str).values,
                             pd.to_numeric(tmp["Y"], errors="coerce").fillna(-999).astype(int).values):
                self.y_by_label[(lt, y)] += 1

        # null counts
        null_series = df.isna().sum()
        for c, k in null_series.to_dict().items():
            self.nulls[c] += int(k)

        # numeric min/max
        for c in NUM_LIKE:
            if c in df.columns:
                col = pd.to_numeric(df[c], errors="coerce")
                if col.notna().any():
                    mn = float(np.nanmin(col.values))
                    mx = float(np.nanmax(col.values))
                    if mn < self.numeric_min[c]:
                        self.numeric_min[c] = mn
                    if mx > self.numeric_max[c]:
                        self.numeric_max[c] = mx

        # range checks for ratios
        rng_cols = []
        for c in ("aggr_ratio_10s","aggr_ratio_15s","bt_dom_3s","bt_dom_5s","bt_dom_10s"):
            if c in df.columns:
                rng_cols.append(c)
        if rng_cols:
            tmp = df[rng_cols].apply(pd.to_numeric, errors="coerce")
            mask = tmp.notna().all(axis=1)
            self.chk_ranges_total += int(mask.sum())
            if mask.any():
                vals = tmp[mask]
                ok = ((vals >= -0.05) & (vals <= 1.05)).all(axis=1).sum()
                self.chk_ranges_ok += int(ok)

        # task check: task == f"{label_type}_{tf}" except legacy
        if {"task","label_type","tf"}.issubset(df.columns):
            lt = df["label_type"].astype(str).values
            tf = df["tf"].astype(str).values
            task = df["task"].astype(str).values
            mask = (lt != "legacy")
            self.chk_task_total += int(mask.sum())
            if mask.any():
                want = np.char.add(np.char.add(lt[mask], "_"), tf[mask])
                self.chk_task_ok += int((task[mask] == want).sum())

        # label logic checks
        if {"label_type","side","Y","y_raw","symbol","year","t","tf"}.issubset(df.columns):
            lt = df["label_type"].astype(str).values
            side = df["side"].astype(str).values
            y = pd.to_numeric(df["Y"], errors="coerce").fillna(-999).astype(int).values
            yraw = pd.to_numeric(df["y_raw"], errors="coerce").fillna(-999).astype(int).values

            # DIR checks (row-level mapping)
            is_dir = (lt == "dir")
            if is_dir.any():
                self.chk_dir_neutral_total += int(is_dir.sum())
                if dir_include_neutral:
                    self.chk_dir_neutral_ok += int(is_dir.sum())
                else:
                    self.chk_dir_neutral_ok += int((yraw[is_dir] != 0).sum())

                self.chk_dir_mapping_total += int(is_dir.sum())
                ok = 0
                for si, yi, yr in zip(side[is_dir], y[is_dir], yraw[is_dir]):
                    if yr == 1:
                        ok += int((si == "buy" and yi == 1) or (si == "sell" and yi == 0))
                    elif yr == -1:
                        ok += int((si == "sell" and yi == 1) or (si == "buy" and yi == 0))
                    elif yr == 0:
                        ok += int(yi == 0)
                    else:
                        ok += 0
                self.chk_dir_mapping_ok += int(ok)

            # GO pair/mapping check (pair-level via cache)
            sym = df["symbol"].astype(str).values
            yr_ = pd.to_numeric(df["year"], errors="coerce").fillna(-999).astype(int).values
            tt = df["t"].astype(str).values
            tfv = df["tf"].astype(str).values
            for s, yy, tts, tff, ltt, sd, yi, yri in zip(sym, yr_, tt, tfv, lt, side, y, yraw):
                if ltt == "go":
                    key = (s, int(yy), tts, tff, "go")
                    self._cache_pair(key, sd, int(yi), int(yri))

            if len(self._pair_cache) > self._pair_cache_max * 0.9:
                self._flush_pairs()

    def finalize(self):
        self._flush_pairs()


class StreamingChecks:
    def __init__(self, max_keys: int = 5_000_000):
        self.key_seen = set()
        self.max_keys = max_keys
        self.dupes_count = 0

    def process_batch(self, df: pd.DataFrame):
        if df is None or df.empty:
            return
        key_cols = ("symbol", "year", "t", "tf", "label_type", "side")
        if all(c in df.columns for c in key_cols):
            for tpl in zip(*(df[c].astype(str).values for c in key_cols)):
                h = hash(tpl)
                if h in self.key_seen:
                    self.dupes_count += 1
                elif len(self.key_seen) < self.max_keys:
                    self.key_seen.add(h)

    def report(self):
        if self.dupes_count > 0:
            _log(f"❌ Doublons (symbol,year,t,tf,label_type,side): {self.dupes_count}")
        else:
            _log("✅ Aucun doublon (symbol,year,t,tf,label_type,side).")


# ----------------------------
# Validation d’un split
# ----------------------------
def _validate_paths(paths: List[str], so: dict, batch_rows: int, dir_include_neutral: bool) -> None:
    if not paths:
        _log("⚠️ Aucun fichier à valider pour ce split.")
        return

    _log(f"→ {len(paths)} fichier(s) à valider (top 20) :")
    for p in paths[:20]:
        _log(f"   - {p}")
    if len(paths) > 20:
        _log(f"   ... (+{len(paths)-20} autres)")

    assert_schema_consistency(paths, so)

    metrics = Metrics()
    schk = StreamingChecks(max_keys=5_000_000)

    for p in paths:
        pf = _open_parquet_file(p, so)
        cols = list(pf.schema.names)
        for batch in pf.iter_batches(batch_size=batch_rows, columns=cols):
            t = pa.Table.from_batches([batch])
            df = t.to_pandas()

            missing_req = [c for c in REQUIRED_COLS if c not in df.columns]
            if missing_req:
                _log(f"⚠️ Missing required columns in {p}: {missing_req}")

            metrics.update_batch(df, dir_include_neutral=dir_include_neutral)
            schk.process_batch(df)

            del df, t, batch
            gc.collect()

    metrics.finalize()

    # Reporting
    print("\n===== RÉSUMÉ (split) =====")
    print(f"Total rows: {metrics.rows:,}")

    print("\n-- Required cols non-null (row-level) --")
    print(_fmt_ratio(metrics.chk_required_ok, metrics.chk_required_total))

    print("\n-- Check task = label_type_tf (non-legacy) --")
    print(_fmt_ratio(metrics.chk_task_ok, metrics.chk_task_total))

    print("\n-- Répartition label_type --")
    for k, v in metrics.label_type_counts.most_common():
        print(f"{k}: {v:,}")

    print("\n-- Répartition task (top 30) --")
    for k, v in metrics.task_counts.most_common(30):
        print(f"{k}: {v:,}")

    print("\n-- Répartition Y (global) --")
    for k, v in metrics.y_counts.most_common():
        print(f"Y={k}: {v:,}")

    print("\n-- Répartition y_raw (global) --")
    for k, v in metrics.yraw_counts.most_common():
        print(f"y_raw={k}: {v:,}")

    print("\n-- Répartition (label_type, Y) --")
    for (lt, y), v in sorted(metrics.y_by_label.items(), key=lambda x: (-x[1], x[0])):
        print(f"{lt:6s}  Y={y:>3d}: {v:,}")

    print("\n-- Répartition TF --")
    for k, v in metrics.tf_counts.most_common():
        print(f"{k}: {v:,}")

    print("\n-- Répartition (label_type, tf) --")
    top = sorted(metrics.tf_by_label.items(), key=lambda kv: -kv[1])[:40]
    for (lt, tf), v in top:
        print(f"{lt:6s} {tf:6s}: {v:,}")
    if len(metrics.tf_by_label) > 40:
        print(f"... (+{len(metrics.tf_by_label)-40} autres)")

    print("\n-- Répartition side --")
    for k, v in metrics.side_counts.most_common():
        print(f"{k}: {v:,}")

    print("\n-- Répartition symbol --")
    for k, v in metrics.symbol_counts.most_common():
        print(f"{k}: {v:,}")

    print("\n-- Années --")
    for k, v in sorted(metrics.year_counts.items()):
        print(f"{k}: {v:,}")

    print("\n-- Taux de nulls par colonne (top 30) --")
    for c, n in metrics.nulls.most_common(30):
        rate = n / metrics.rows if metrics.rows else 0.0
        print(f"{c:30s} {n:>10,d}  ({rate:.2%})")

    print("\n-- Bornes numériques (min .. max) --")
    for c in NUM_LIKE:
        mn = metrics.numeric_min.get(c, math.inf)
        mx = metrics.numeric_max.get(c, -math.inf)
        if mn is not math.inf and mx is not -math.inf:
            print(f"{c:30s} {mn:>12.6g} .. {mx:>12.6g}")

    # Stage2 semantic checks
    print("\n-- Check GO: paires buy+sell par (symbol,year,t,tf,label_type) --")
    print(_fmt_ratio(metrics.chk_go_pair_ok, metrics.chk_go_pair_total))

    print("\n-- Check GO: mapping (y_raw -> Y buy/sell) --")
    print(_fmt_ratio(metrics.chk_go_mapping_ok, metrics.chk_go_mapping_total))

    print("\n-- Check DIR: neutral y_raw==0 --")
    if dir_include_neutral:
        print("DIR neutral autorisé (dir_include_neutral=ON) -> PASS (by config)")
    else:
        print(_fmt_ratio(metrics.chk_dir_neutral_ok, metrics.chk_dir_neutral_total))

    print("\n-- Check DIR: mapping (y_raw -> Y row-level) --")
    print(_fmt_ratio(metrics.chk_dir_mapping_ok, metrics.chk_dir_mapping_total))

    if metrics.chk_ranges_total > 0:
        print("\n-- Check ranges (ratios ~ [0,1]) --")
        print(_fmt_ratio(metrics.chk_ranges_ok, metrics.chk_ranges_total))
    else:
        print("\n-- Check ranges (ratios ~ [0,1]) --")
        print("NA (colonnes ratio absentes)")

    # Quick alarms
    alarms = []
    tot_y = sum(v for k, v in metrics.y_counts.items() if pd.notna(k))
    maj_prop = max((v / tot_y) for _, v in metrics.y_counts.items()) if tot_y else 0
    if maj_prop > 0.95:
        alarms.append(f"Distribution Y très déséquilibrée (classe majoritaire ≈ {maj_prop:.1%}).")

    if "spread_bps_entry" in metrics.numeric_min and metrics.numeric_min["spread_bps_entry"] < -1e-6:
        alarms.append("spread_bps_entry < 0 détecté (devrait être ≥ 0).")
    if "spread_bps_top" in metrics.numeric_min and metrics.numeric_min["spread_bps_top"] < -1e-6:
        alarms.append("spread_bps_top < 0 détecté (devrait être ≥ 0).")

    for col in ("aggr_ratio_10s_side", "aggr_ratio_15s_side", "bt_dom_3s_side", "bt_dom_10s_side"):
        if col in metrics.numeric_min:
            mn, mx = metrics.numeric_min[col], metrics.numeric_max[col]
            if mn < -1.2 or mx > 1.2:
                alarms.append(f"{col} hors [-1,1] (min={mn:.3g}, max={mx:.3g}) — vérifier mapping side.")

    if alarms:
        print("\n-- ALARMES / REMARQUES --")
        for a in alarms:
            print("•", a)

    print("\n-- Contrôles avancés (streaming) --")
    schk.report()
    print("\n✅ Validation (split) terminée.\n")


# ----------------------------
# Orchestration multi-splits
# ----------------------------
def run_validate(
    src_root: str,
    split: str,
    symbols: Optional[List[str]],
    years: Optional[List[int]],
    s3_region: Optional[str],
    s3_anon: bool,
    s3_requester_pays: bool,
    batch_rows: int,
    dir_include_neutral: bool,
    debug: bool
):
    so = _so(s3_region, s3_anon, s3_requester_pays)

    # Robustesse: argparse peut renvoyer une string si default mal défini
    if symbols is None:
        symbols = None
    elif isinstance(symbols, str):
        symbols = [symbols]

    if split == "all":
        # "all" = dossier réel stage2/all/...
        splits = ["all"]
    elif split == "all_dir":
        # compat legacy: pareil que "all"
        splits = ["all"]
    elif split == "tvt":
        # nouveau mode: train/val/test
        splits = ["train", "val", "test"]
    else:
        splits = [split]

    for sp in splits:
        print("=" * 80)
        print(f"🔎 Split: {sp}")
        print("=" * 80)

        paths = _glob_split_paths(src_root, sp, so, symbols, years, debug=debug)
        if not paths:
            _log(f"⚠️ Aucun fichier trouvé pour split={sp}")
        else:
            _log(f"✅ {len(paths)} fichier(s) trouvé(s) pour split={sp}")
        _validate_paths(paths, so, batch_rows, dir_include_neutral=dir_include_neutral)


# ----------------------------
# CLI
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser("Validate Stage2 v3.1 parquet(s) on S3 (train/val/test/all)")
    p.add_argument("--src-root", default="s3://tradebot-config-tokyo/data/stage2")
    p.add_argument("--split", choices=["all", "tvt", "train", "val", "test"],
                   default="all",
                   help="all = dossier 'all' ; tvt = train/val/test ; sinon split unique."
    )
    p.add_argument("--symbols", nargs="*", default=["BTCUSDT"], help="Ex: BTCUSDT ETHUSDT APTUSDT")
    p.add_argument("--years", nargs="*", type=int, default=None, help="Ex: 2023 2024 2025")
    p.add_argument("--s3-region", default="ap-northeast-1")
    p.add_argument("--s3-anon", action="store_true")
    p.add_argument("--s3-requester-pays", action="store_true")
    p.add_argument("--batch-rows", type=int, default=200_000,
                   help="Taille batch Arrow (mémoire/performances).")
    p.add_argument("--dir-include-neutral", action="store_true",
                   help="Autorise y_raw==0 dans label_type=dir (sinon c'est une erreur).")
    p.add_argument("--debug", action="store_true")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_validate(
        args.src_root, args.split,
        args.symbols, args.years,
        args.s3_region, args.s3_anon, args.s3_requester_pays,
        args.batch_rows,
        dir_include_neutral=bool(args.dir_include_neutral),
        debug=args.debug
    )