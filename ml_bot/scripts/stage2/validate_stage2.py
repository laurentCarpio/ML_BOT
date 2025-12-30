#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
validate_stage2.py — Validation Stage2 v4.1 (STRICT, non rétro-compatible), memory-safe.

Layout UNIQUE supporté:
  <src_root>/<split>/<dataset>/<SYMBOL>/<YEAR>/parts/part-*.parquet
  où dataset ∈ {go, dir}
  ex:
    s3://tradebot-config-tokyo/data/stage2/all/go/BTCUSDT/2024/parts/part-00000.parquet
    s3://tradebot-config-tokyo/data/stage2/all/dir/BTCUSDT/2024/parts/part-00000.parquet

Ce validateur (strict v4.1):
- vérifie colonnes requises (go/dir) — STRICT: y_go + y_dir présents dans tous les datasets
- vérifie event_id non-null + unicité (streaming)
- vérifie formule event_id = row_id|task|tf (tf normalisé lower/strip)
- vérifie task == dataset ("go" ou "dir") (task normalisé)
- vérifie domaines y_go (0/1) + y_dir (-1/0/+1 pour GO ; -1/+1 pour DIR)
- checks de ranges (ratios [0,1], etc.)
"""

from __future__ import annotations

import argparse
import re
import math
import gc
import hashlib
from collections import Counter, defaultdict
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import fsspec
from botocore.exceptions import ClientError


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

def _ensure_proto(fs, p: str) -> str:
    proto = fs.protocol[0] if isinstance(fs.protocol, (list, tuple)) else fs.protocol
    if not re.match(r"^[a-z0-9]+://", str(p)):
        return f"{proto}://{p}"
    return str(p)

def _norm_str(x) -> str:
    if isinstance(x, (bytes, bytearray)):
        return x.decode("utf-8", "ignore")
    return str(x)

def _norm_tf(x) -> str:
    return _norm_str(x).strip().lower()

def _norm_task(x) -> str:
    return _norm_str(x).strip().lower()


# ----------------------------
# Découverte des fichiers (STRICT: un seul pattern)
# ----------------------------
def _glob_one(pat: str, so: dict, debug: bool = False) -> List[str]:
    if debug:
        _log(f"[glob] try: {pat}")
    try:
        fs, _, paths = fsspec.get_fs_token_paths(pat, storage_options=so)
        out = [_ensure_proto(fs, p) for p in sorted(paths)]
        if debug:
            _log(f"[glob]  -> {len(out)} match(es)")
        return out
    except PermissionError:
        _log(f"❌ AccessDenied glob: {pat}")
        return []
    except ClientError as e:
        _log(f"❌ ClientError glob: {pat} — {e}")
        return []
    except Exception as e:
        _log(f"❌ glob error: {pat} — {e!r}")
        return []

def _glob_split_paths_strict(
    src_root: str,
    dataset: str,
    split: str,
    so: dict,
    symbols: Optional[List[str]],
    years: Optional[List[int]],
    debug: bool = False
) -> List[str]:
    """
    STRICT pattern:
      <src_root>/<split>/<dataset>/<sym>/<year>/parts/part-*.parquet
    """
    src_root = src_root.rstrip("/")
    syms = symbols or ["*"]
    yrs  = [str(y) for y in (years or ["*"])]

    patterns = [
        f"{src_root}/{split}/{dataset}/{s}/{y}/parts/part-*.parquet"
        for s in syms for y in yrs
    ]
    paths: List[str] = []
    for pat in patterns:
        paths.extend(_glob_one(pat, so, debug=debug))
    return sorted(set(paths))


# ----------------------------
# Schéma homogène
# ----------------------------
def assert_schema_consistency(paths: List[str], so: dict) -> None:
    if not paths:
        return
    pf0 = _open_parquet_file(paths[0], so)
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
        raise ValueError("Schema/types inconsistents entre parts.")
    else:
        _log("✅ Schéma/types Arrow homogènes sur toutes les parts.")

def _fmt_ratio(ok: int, tot: int) -> str:
    return f"{ok}/{tot} = {ok/tot:.2%}" if tot else "0/0"


# ----------------------------
# Unicité streaming (event_id)
# ----------------------------
def _hash64_blake2b(s: str) -> int:
    # digest_size=8 => 64-bit
    h = hashlib.blake2b(s.encode("utf-8", "ignore"), digest_size=8).digest()
    return int.from_bytes(h, byteorder="big", signed=False)

class UniqueEventIdChecks:
    def __init__(self, max_keys: int = 5_000_000):
        self.max_keys = int(max_keys)
        self.seen = set()
        self.dup = 0
        self.null = 0
        self.sample_dups = []

    def process_batch(self, df: pd.DataFrame):
        if df is None or df.empty or "event_id" not in df.columns:
            return

        raw = df["event_id"]
        is_null = raw.isna()
        self.null += int(is_null.sum())

        ev2 = raw[~is_null].astype(str).values
        for s in ev2:
            h = _hash64_blake2b(str(s))
            if h in self.seen:
                self.dup += 1
                if len(self.sample_dups) < 10:
                    self.sample_dups.append(str(s))
            else:
                if len(self.seen) < self.max_keys:
                    self.seen.add(h)

    def report(self):
        if self.null > 0:
            _log(f"❌ event_id NULL: {self.null}")
        else:
            _log("✅ event_id non-null (aucun NULL détecté).")

        if self.dup > 0:
            _log(f"❌ event_id dupliqués: {self.dup}")
            if self.sample_dups:
                _log("   exemples dup:")
                for s in self.sample_dups:
                    _log(f"   - {s}")
        else:
            _log("✅ event_id uniques (aucun doublon détecté).")


# ----------------------------
# Metrics + checks
# ----------------------------
class Metrics:
    def __init__(self):
        self.rows = 0
        self.nulls = Counter()
        self.numeric_min = defaultdict(lambda: math.inf)
        self.numeric_max = defaultdict(lambda: -math.inf)

        self.tf_counts = Counter()
        self.task_counts = Counter()
        self.year_counts = Counter()
        self.symbol_counts = Counter()

        self.ygo_counts = Counter()
        self.ydir_counts = Counter()

        # séparé: structure vs labels
        self.chk_struct_ok = 0
        self.chk_struct_total = 0

        self.chk_labels_ok = 0
        self.chk_labels_total = 0

        self.chk_task_ok = 0
        self.chk_task_total = 0

        self.chk_event_formula_ok = 0
        self.chk_event_formula_total = 0

        self.chk_ygo_ok = 0
        self.chk_ygo_total = 0

        self.chk_ydir_ok = 0
        self.chk_ydir_total = 0

        self.chk_ranges_ok = 0
        self.chk_ranges_total = 0

    def update_batch(
        self,
        df: pd.DataFrame,
        required_structure: List[str],
        required_labels: List[str],
        num_like: List[str],
        dataset: str,
    ):
        if df is None or df.empty:
            return
        n = len(df)
        self.rows += n

        # required non-null: structure
        self.chk_struct_total += n
        ok_struct = df[required_structure].notna().all(axis=1).sum()
        self.chk_struct_ok += int(ok_struct)

        # required non-null: labels
        self.chk_labels_total += n
        ok_labels = df[required_labels].notna().all(axis=1).sum()
        self.chk_labels_ok += int(ok_labels)

        # counts
        for c in ["tf", "symbol", "year", "task"]:
            if c in df.columns:
                vc = df[c].value_counts(dropna=False)
                if c == "tf":
                    self.tf_counts.update(vc.to_dict())
                elif c == "symbol":
                    self.symbol_counts.update(vc.to_dict())
                elif c == "year":
                    self.year_counts.update(vc.to_dict())
                elif c == "task":
                    self.task_counts.update(vc.to_dict())

        if "y_go" in df.columns:
            self.ygo_counts.update(df["y_go"].value_counts(dropna=False).to_dict())
        if "y_dir" in df.columns:
            self.ydir_counts.update(df["y_dir"].value_counts(dropna=False).to_dict())

        # null counts (toutes colonnes, utile pour audit)
        null_series = df.isna().sum()
        for c, k in null_series.to_dict().items():
            self.nulls[c] += int(k)

        # numeric min/max
        for c in num_like:
            if c in df.columns:
                col = pd.to_numeric(df[c], errors="coerce")
                if col.notna().any():
                    mn = float(np.nanmin(col.values))
                    mx = float(np.nanmax(col.values))
                    if mn < self.numeric_min[c]:
                        self.numeric_min[c] = mn
                    if mx > self.numeric_max[c]:
                        self.numeric_max[c] = mx

        # task strict: must equal dataset ("go" or "dir") — normalize
        if "task" in df.columns:
            self.chk_task_total += n
            task = df["task"].map(_norm_task).values
            self.chk_task_ok += int((task == str(dataset).lower()).sum())

        # event_id strict formula: row_id|task|tf (tf normalized)
        if {"event_id", "row_id", "task", "tf"}.issubset(df.columns):
            raw_ok = (~df["event_id"].isna()) & (~df["row_id"].isna()) & (~df["task"].isna()) & (~df["tf"].isna())
            self.chk_event_formula_total += int(raw_ok.sum())
            if raw_ok.any():
                rid = df.loc[raw_ok, "row_id"].astype(str).values
                taskv = df.loc[raw_ok, "task"].map(_norm_task).astype(str).values
                tfv = df.loc[raw_ok, "tf"].map(_norm_tf).astype(str).values
                want = np.char.add(np.char.add(np.char.add(rid, "|"), taskv), np.char.add("|", tfv))
                eid = df.loc[raw_ok, "event_id"].astype(str).values
                self.chk_event_formula_ok += int((eid == want).sum())

        # y_go strict
        if "y_go" in df.columns:
            y = pd.to_numeric(df["y_go"], errors="coerce")
            ok_mask = y.notna()
            self.chk_ygo_total += int(ok_mask.sum())
            self.chk_ygo_ok += int(((y[ok_mask] == 0) | (y[ok_mask] == 1)).sum())

        # y_dir strict depends on dataset
        if "y_dir" in df.columns:
            y = pd.to_numeric(df["y_dir"], errors="coerce")
            ok_mask = y.notna()
            self.chk_ydir_total += int(ok_mask.sum())
            if dataset == "dir":
                self.chk_ydir_ok += int(((y[ok_mask] == -1) | (y[ok_mask] == 1)).sum())
            else:
                self.chk_ydir_ok += int(((y[ok_mask] == -1) | (y[ok_mask] == 0) | (y[ok_mask] == 1)).sum())

        # range checks (ratios in [0,1])
        ranges = {
            "aggr_ratio_10s": (0.0, 1.0),
            "aggr_ratio_15s": (0.0, 1.0),
            "bt_dom_3s": (0.0, 1.0),
            "bt_dom_5s": (0.0, 1.0),
            "bt_dom_10s": (0.0, 1.0),
            "wall_opp_share_5_buy": (0.0, 1.0),
            "wall_opp_share_5_sell": (0.0, 1.0),
            "wall_opp_share_15_buy": (0.0, 1.0),
            "wall_opp_share_15_sell": (0.0, 1.0),
        }
        rng_cols = [c for c in ranges.keys() if c in df.columns]
        if rng_cols:
            tmp = df[rng_cols].apply(pd.to_numeric, errors="coerce")
            mask = tmp.notna().all(axis=1)
            self.chk_ranges_total += int(mask.sum())
            if mask.any():
                vals = tmp[mask]
                ok_row = np.ones(len(vals), dtype=bool)
                for c in rng_cols:
                    lo, hi = ranges[c]
                    ok_row &= (vals[c].values >= lo - 1e-6) & (vals[c].values <= hi + 1e-6)
                self.chk_ranges_ok += int(ok_row.sum())

# ----------------------------
# Validation
# ----------------------------
def _filter_rows(df: pd.DataFrame, symbols: Optional[List[str]], years: Optional[List[int]]) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if symbols and "symbol" in df.columns:
        sset = set(map(str, symbols))
        df = df[df["symbol"].astype(str).isin(sset)]
    if years and "year" in df.columns:
        yset = set(int(y) for y in years)
        yy = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
        df = df[yy.isin(yset)]
    return df

def _validate_paths(
    paths: List[str],
    so: dict,
    batch_rows: int,
    dataset: str,
    symbols: Optional[List[str]],
    years: Optional[List[int]],
):
    if not paths:
        _log("⚠️ Aucun fichier à valider.")
        return

    _log(f"→ {len(paths)} fichier(s) à valider (top 20):")
    for p in paths[:20]:
        _log(f"   - {p}")
    if len(paths) > 20:
        _log(f"   ... (+{len(paths)-20} autres)")

    assert_schema_consistency(paths, so)


    # Stage2 = dataset integrity (doit être non-null)
    REQUIRED_STRUCTURE = [
        "t", "symbol", "year", "tf", "row_id", "task", "event_id",
    ]

    # Labels (souvent requis pour train/val/test, mais ce sont des cibles, pas de la structure)
    REQUIRED_LABELS = [
        "y_go", "y_dir",
    ]

    # Features (NE PAS rendre required au niveau Stage2)
    FEATURE_COLS = [
        "bid_entry", "ask_entry", "spread_bps_entry", "atr_bps",
    ]

    # Compat: si ton code attend une variable REQUIRED
    REQUIRED = REQUIRED_STRUCTURE + REQUIRED_LABELS

    NUM_LIKE = [
        "bid_entry","ask_entry","spread_bps_entry","atr_bps",
        "mid_t","spread_bps_top",
        "obi_5","obi_15","microprice_bias",
        "slope_bid_5","slope_ask_5","slope_bid_15","slope_ask_15",
        "quote_churn_10s","ret_stdev_1s_10s_bps",
        "aggr_ratio_10s","aggr_ratio_15s",
        "bt_dom_3s","bt_dom_5s","bt_dom_10s",
        "obi_5_side_buy","obi_5_side_sell",
        "obi_15_side_buy","obi_15_side_sell",
        "microprice_bias_side_buy","microprice_bias_side_sell",
        "aggr_ratio_10s_side_buy","aggr_ratio_10s_side_sell",
        "aggr_ratio_15s_side_buy","aggr_ratio_15s_side_sell",
        "bt_dom_3s_side_buy","bt_dom_3s_side_sell",
        "bt_dom_10s_side_buy","bt_dom_10s_side_sell",
        "wall_opp_share_5_buy","wall_opp_share_5_sell",
        "wall_opp_share_15_buy","wall_opp_share_15_sell",
        "cum_depth_within_5bps_opp_buy","cum_depth_within_5bps_opp_sell",
        "cum_depth_within_10bps_opp_buy","cum_depth_within_10bps_opp_sell",
        "slope_opp_5_buy","slope_opp_5_sell",
        "slope_opp_15_buy","slope_opp_15_sell",
        "bb_width","bb_width_pctl","lf_bb_width_pct",
        "atr_percentile","lf_atr_rank_30m","atr_pct_rank_30m","adx",
    ]

    metrics = Metrics()
    uniq = UniqueEventIdChecks(max_keys=5_000_000)

    for p in paths:
        pf = _open_parquet_file(p, so)
        cols = list(pf.schema.names)

        for batch in pf.iter_batches(batch_size=batch_rows, columns=cols):
            t = pa.Table.from_batches([batch])
            df = t.to_pandas()

            df = _filter_rows(df, symbols=symbols, years=years)
            if df.empty:
                del df, t, batch
                gc.collect()
                continue

            missing = [c for c in (REQUIRED_STRUCTURE + REQUIRED_LABELS) if c not in df.columns]
            if missing:
                raise ValueError(f"Missing required columns in {p}: {missing}")

            # strict task check early (normalized)
            task_norm = df["task"].map(_norm_task)
            bad_task = task_norm != str(dataset).lower()
            if bad_task.any():
                ex = df.loc[bad_task, ["symbol","year","tf","row_id","task","event_id"]].head(5)
                raise ValueError(f"Bad task values (expected task='{dataset}') in {p}. Examples:\n{ex}")

            metrics.update_batch(df,
                                 required_structure=REQUIRED_STRUCTURE,
                                 required_labels=REQUIRED_LABELS,
                                 num_like=NUM_LIKE,
                                 dataset=dataset,
                                 )
            uniq.process_batch(df)

            del df, t, batch
            gc.collect()

    print("\n===== RÉSUMÉ =====")
    print(f"Dataset: {dataset}")
    print(f"Total rows: {metrics.rows:,}")

    print("\n-- Required cols non-null (structure, row-level) --")
    print(_fmt_ratio(metrics.chk_struct_ok, metrics.chk_struct_total))

    print("\n-- Required cols non-null (labels, row-level) --")
    print(_fmt_ratio(metrics.chk_labels_ok, metrics.chk_labels_total))

    print("\n-- Feature nulls (info only) --")
    for c in FEATURE_COLS:
        if c in metrics.nulls:
            n = metrics.nulls[c]
            rate = n / metrics.rows if metrics.rows else 0.0
            print(f"{c:30s} {n:>10,d}  ({rate:.2%})")

    print("\n-- Check task == dataset --")
    print(_fmt_ratio(metrics.chk_task_ok, metrics.chk_task_total))

    print("\n-- Check event_id formula (row_id|task|tf) --")
    print(_fmt_ratio(metrics.chk_event_formula_ok, metrics.chk_event_formula_total))

    print("\n-- Check y_go ∈ {0,1} --")
    print(_fmt_ratio(metrics.chk_ygo_ok, metrics.chk_ygo_total))
    print("\n-- Répartition y_go --")
    for k, v in metrics.ygo_counts.most_common():
        print(f"y_go={k}: {v:,}")

    print("\n-- Check y_dir domain --")
    print(_fmt_ratio(metrics.chk_ydir_ok, metrics.chk_ydir_total))
    print("\n-- Répartition y_dir --")
    for k, v in metrics.ydir_counts.most_common():
        print(f"y_dir={k}: {v:,}")

    print("\n-- Répartition TF --")
    for k, v in metrics.tf_counts.most_common():
        print(f"{k}: {v:,}")

    print("\n-- Répartition task --")
    for k, v in metrics.task_counts.most_common():
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
    for c in sorted(metrics.numeric_min.keys()):
        mn = metrics.numeric_min[c]
        mx = metrics.numeric_max[c]
        if mn is not math.inf and mx is not -math.inf:
            print(f"{c:30s} {mn:>12.6g} .. {mx:>12.6g}")

    if metrics.chk_ranges_total > 0:
        print("\n-- Check ranges (ratios ~ [0,1]) --")
        print(_fmt_ratio(metrics.chk_ranges_ok, metrics.chk_ranges_total))
    else:
        print("\n-- Check ranges (ratios ~ [0,1]) --")
        print("NA (colonnes ratio absentes)")

    print("\n-- Checks event_id (unicité + non-null) --")
    uniq.report()

    print("\n✅ Validation terminée.\n")


def run_validate(
    src_root: str,
    dataset: str,
    split: str,
    symbols: Optional[List[str]],
    years: Optional[List[int]],
    s3_region: Optional[str],
    s3_anon: bool,
    s3_requester_pays: bool,
    batch_rows: int,
    debug: bool
):
    so = _so(s3_region, s3_anon, s3_requester_pays)

    if split == "all":
        splits = ["all"]
    elif split == "tvt":
        splits = ["train", "val", "test"]
    else:
        splits = [split]

    for sp in splits:
        print("=" * 80)
        print(f"🔎 Dataset={dataset} | Split={sp}")
        print("=" * 80)

        paths = _glob_split_paths_strict(src_root, dataset, sp, so, symbols, years, debug=debug)
        if not paths:
            _log(f"⚠️ Aucun fichier trouvé (dataset={dataset}, split={sp})")
            continue

        _log(f"✅ {len(paths)} fichier(s) trouvé(s)")

        _validate_paths(
            paths, so,
            batch_rows=batch_rows,
            dataset=dataset,
            symbols=symbols,
            years=years,
        )


def parse_args():
    p = argparse.ArgumentParser("Validate Stage2 v4.1 (STRICT) — GO wide / DIR wide")
    p.add_argument("--src-root", default="s3://tradebot-config-tokyo/data/stage2",
                   help="Root. STRICT: <root>/<split>/<dataset>/<sym>/<year>/parts/part-*.parquet")
    p.add_argument("--dataset", choices=["go", "dir"], required=True)
    p.add_argument("--split", choices=["all", "tvt", "train", "val", "test"], default="all")
    p.add_argument("--symbols", nargs="*", default=None)   # IMPORTANT: pas de filtre par défaut
    p.add_argument("--years", nargs="*", type=int, default=None)
    p.add_argument("--s3-region", default="ap-northeast-1")
    p.add_argument("--s3-anon", action="store_true")
    p.add_argument("--s3-requester-pays", action="store_true")
    p.add_argument("--batch-rows", type=int, default=200_000)
    p.add_argument("--debug", action="store_true")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_validate(
        src_root=args.src_root,
        dataset=args.dataset,
        split=args.split,
        symbols=args.symbols,
        years=args.years,
        s3_region=args.s3_region,
        s3_anon=args.s3_anon,
        s3_requester_pays=args.s3_requester_pays,
        batch_rows=args.batch_rows,
        debug=args.debug,
    )