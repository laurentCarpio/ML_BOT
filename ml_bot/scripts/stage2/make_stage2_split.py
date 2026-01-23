#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_stage2_split.py — Stage2 v4.1
SRC:  stage2/all/{go,dir}/{SYMBOL}/{YEAR}/parts/part-*.parquet
DST:  stage2/split/{train,val,test}/{go,dir}/{SYMBOL}/{YEAR}/parts/part-*.parquet
META: stage2/_meta/split_cutoffs.json  (partagé pour go+dir)

- Fit des cutoffs une seule fois (sur go par défaut)
- Split GO + DIR avec les mêmes cutoffs
- Validation streaming:
  - task == dataset ("go" ou "dir")
  - event_id == f"{row_id}|{task}|{tf}"
  - event_id unique + non-null
"""

from __future__ import annotations

import argparse
import time
import json
import random
import hashlib
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List
from collections import defaultdict

import numpy as np
import pandas as pd

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pyarrow.fs as pafs

# ----------------------------
# S3 helpers
# ----------------------------
def _strip_s3(uri: str) -> str:
    uri = str(uri)
    assert uri.startswith("s3://"), f"expected s3://..., got {uri}"
    return uri[len("s3://"):]

def _fs(region: Optional[str]) -> pafs.S3FileSystem:
    return pafs.S3FileSystem(region=region) if region else pafs.S3FileSystem()

def dataset_stage2_all(fs: pafs.S3FileSystem, src_stage2_all_root: str, dataset_name: str) -> ds.Dataset:
    """
    src_stage2_all_root attendu: s3://.../data/stage2/all
    Structure:
      all/<dataset>/<SYMBOL>/<YEAR>/parts/*.parquet
      dataset ∈ {go, dir}

    NOTE: pas de partitioning hive (folders non key=value).
    """
    base = _strip_s3(src_stage2_all_root.rstrip("/"))
    base = f"{base}/{dataset_name}"
    return ds.dataset(base, filesystem=fs, format="parquet", partitioning="hive")

def _write_parquet(fs: pafs.S3FileSystem, path_s3: str, df: pd.DataFrame):
    table = pa.Table.from_pandas(df, preserve_index=False)
    with fs.open_output_stream(_strip_s3(path_s3)) as out:
        pq.write_table(table, out, compression="zstd")

def _h64(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8","ignore"), digest_size=8).digest(), "big")

# ----------------------------
# Normalizers
# ----------------------------
def _norm_str(x) -> str:
    if isinstance(x, (bytes, bytearray)):
        return x.decode("utf-8", "ignore")
    return str(x)

def _norm_tf(x) -> str:
    return _norm_str(x).strip().lower()

# ----------------------------
# Streaming validators
# ----------------------------
class UniqueEventIdChecks:
    """Unicité event_id en streaming (hash-set capé)."""
    def __init__(self, max_keys: int = 5_000_000):
        self.max_keys = int(max_keys)
        self.seen = set()
        self.dup = 0
        self.null = 0
        self.sample_dups = []

    def process(self, event_id_col: pd.Series):
        raw = event_id_col
        is_null = raw.isna()
        self.null += int(is_null.sum())

        ev = raw[~is_null].astype(str).values
        for s in ev:
            h = _h64(s)
            if h in self.seen:
                self.dup += 1
                if len(self.sample_dups) < 10:
                    self.sample_dups.append(s)
            else:
                if len(self.seen) < self.max_keys:
                    self.seen.add(h)

    def finalize_or_raise(self, tag: str):
        if self.null > 0:
            raise ValueError(f"[{tag}] event_id NULL détecté: n={self.null}")
        if self.dup > 0:
            msg = f"[{tag}] event_id dupliqués détectés: n={self.dup}"
            if self.sample_dups:
                msg += f" exemples={self.sample_dups}"
            raise ValueError(msg)

class EventFormulaChecks:
    """
    Stage2 v4.1 STRICT:
      event_id == f"{row_id}|{task}|{tf}"
    Et row_id/task/tf/event_id non-null.
    """
    def __init__(self):
        self.total = 0
        self.bad = 0
        self.bad_examples = []

        self.rowid_null = 0
        self.task_null = 0
        self.tf_null = 0
        self.event_null = 0

    def process(self, df: pd.DataFrame):
        self.rowid_null += int(df["row_id"].isna().sum())
        self.task_null  += int(df["task"].isna().sum())
        self.tf_null    += int(df["tf"].isna().sum())
        self.event_null += int(df["event_id"].isna().sum())

        ok = (~df["row_id"].isna()) & (~df["task"].isna()) & (~df["tf"].isna()) & (~df["event_id"].isna())
        self.total += int(ok.sum())
        if not ok.any():
            return

        rid  = df.loc[ok, "row_id"].astype(str).values
        task = df.loc[ok, "task"].astype(str).str.strip().str.lower().values
        tf   = df.loc[ok, "tf"].astype(str).map(_norm_tf).values
        eid  = df.loc[ok, "event_id"].astype(str).values

        want = np.char.add(np.char.add(np.char.add(rid, "|"), task), np.char.add("|", tf))

        bad_mask = (eid != want)
        nb = int(bad_mask.sum())
        if nb > 0:
            self.bad += nb
            if len(self.bad_examples) < 10:
                idx = np.where(bad_mask)[0][: (10 - len(self.bad_examples))]
                for j in idx:
                    self.bad_examples.append((rid[j], task[j], tf[j], eid[j], want[j]))

    def finalize_or_raise(self, tag: str):
        if self.rowid_null > 0:
            raise ValueError(f"[{tag}] row_id NULL détecté: n={self.rowid_null}")
        if self.task_null > 0:
            raise ValueError(f"[{tag}] task NULL détecté: n={self.task_null}")
        if self.tf_null > 0:
            raise ValueError(f"[{tag}] tf NULL détecté: n={self.tf_null}")
        if self.event_null > 0:
            raise ValueError(f"[{tag}] event_id NULL détecté: n={self.event_null}")
        if self.bad > 0:
            msg = f"[{tag}] event_id formula mismatch: n={self.bad}/{self.total}"
            if self.bad_examples:
                msg += " exemples=" + repr(self.bad_examples)
            raise ValueError(msg)

class TaskChecks:
    """Stage2 v4.1: task doit être exactement 'go' ou 'dir' selon le dataset."""
    def __init__(self, dataset_name: str):
        self.dataset_name = str(dataset_name)
        self.total = 0
        self.bad = 0
        self.bad_examples = []

    def process(self, df: pd.DataFrame):
        if "task" not in df.columns:
            raise ValueError("missing task column")
        task = df["task"].astype(str).str.strip().str.lower().values
        self.total += len(task)
        bad_mask = (task != self.dataset_name)
        nb = int(bad_mask.sum())
        if nb > 0:
            self.bad += nb
            if len(self.bad_examples) < 10:
                ex = df.loc[bad_mask, ["row_id", "tf", "task", "event_id"]].head(10 - len(self.bad_examples))
                for row in ex.itertuples(index=False):
                    self.bad_examples.append(tuple(row))

    def finalize_or_raise(self, tag: str):
        if self.bad > 0:
            raise ValueError(f"[{tag}] task mismatch: n={self.bad}/{self.total} exemples={self.bad_examples}")

# ----------------------------
# Split policy
# ----------------------------
@dataclass(frozen=True)
class SplitPolicy:
    q_train_end: float = 0.70
    q_val_end: float = 0.85
    per_tf: bool = True

def fit_time_cutoffs(
    ds_all: ds.Dataset,
    policy: SplitPolicy,
    symbols: Optional[List[str]] = None,
    tfs: Optional[List[str]] = None,
    batch_size: int = 400_000,
    sample_per_key: int = 200_000,
    seed: int = 1337,
) -> Dict[str, Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Fit cutoffs sur la distribution de t, par tf si per_tf=True.
    (On s'en sert ensuite pour splitter go ET dir identiquement.)
    """
    rng = random.Random(seed)

    schema = ds_all.schema
    cols = set(schema.names)
    need = ["t", "tf"]
    missing = [c for c in need if c not in cols]
    if missing:
        raise ValueError(f"Stage2(all) manque des colonnes pour fitter le split: {missing}")

    expr = None
    if symbols and "symbol" in cols:
        expr = ds.field("symbol").isin([str(s) for s in symbols])
    if tfs and "tf" in cols:
        e2 = ds.field("tf").isin([str(x).lower().strip() for x in tfs])
        expr = e2 if expr is None else (expr & e2)

    scanner = ds_all.scanner(columns=need + (["symbol"] if "symbol" in cols else []),
                             filter=expr, batch_size=batch_size)

    res = defaultdict(list)
    seen = defaultdict(int)

    def reservoir_add(key: str, x: int):
        seen[key] += 1
        r = res[key]
        if len(r) < sample_per_key:
            r.append(x)
            return
        j = rng.randrange(seen[key])
        if j < sample_per_key:
            r[j] = x

    for b in scanner.to_batches():
        df = b.to_pandas(types_mapper=pd.ArrowDtype)
        if df.empty:
            continue

        t = pd.to_datetime(df["t"], utc=True, errors="coerce")
        ok = t.notna()
        if not ok.any():
            continue

        tns = t.loc[ok].astype("int64").to_numpy()
        tfv = df.loc[ok, "tf"].map(_norm_tf).to_numpy(dtype=object)

        if policy.per_tf:
            for i in range(len(tns)):
                reservoir_add(str(tfv[i]), int(tns[i]))
        else:
            for i in range(len(tns)):
                reservoir_add("__global__", int(tns[i]))

    if not res:
        raise RuntimeError("fit_time_cutoffs: aucun échantillon après filtres.")

    def q_ns(arr: np.ndarray, q: float) -> int:
        return int(np.quantile(arr.astype("float64"), q))

    out: Dict[str, Tuple[pd.Timestamp, pd.Timestamp]] = {}
    for key, xs in res.items():
        arr = np.array(xs, dtype="int64")
        tr = pd.to_datetime(q_ns(arr, policy.q_train_end), utc=True)
        va = pd.to_datetime(q_ns(arr, policy.q_val_end), utc=True)
        out[str(key)] = (tr, va)

    return out

def save_cutoffs_json(fs: pafs.S3FileSystem, dst_stage2_root: str, cutoffs: Dict[str, Tuple[pd.Timestamp, pd.Timestamp]]):
    meta_root = f"{dst_stage2_root.rstrip('/')}/_meta"

    payload = {k: {"train_end": str(v[0]), "val_end": str(v[1])} for k, v in cutoffs.items()}
    with fs.open_output_stream(_strip_s3(f"{meta_root}/split_cutoffs.json")) as f:
        f.write(json.dumps(payload, indent=2).encode("utf-8"))

def assign_split(df: pd.DataFrame, cutoffs: dict) -> np.ndarray:
    # IMPORTANT: df["t"] peut être timestamp[ns][pyarrow]
    tt = df["t"].to_numpy(dtype="datetime64[ns]", na_value=np.datetime64("NaT"))

    out = np.full(len(df), "test", dtype=object)

    tf_col = df["tf"].astype(str).str.lower().str.strip().to_numpy()

    def _as_dt64(x) -> np.datetime64:
        if isinstance(x, np.datetime64):
            return x.astype("datetime64[ns]")
        if isinstance(x, pd.Timestamp):
            return x.to_datetime64()
        return pd.to_datetime(x, utc=True, errors="coerce").to_datetime64()

    if "__global__" in cutoffs:
        tr, va = cutoffs["__global__"]
        tr64, va64 = _as_dt64(tr), _as_dt64(va)
        m_tr = tt <= tr64
        m_va = (tt > tr64) & (tt <= va64)
        out[m_tr] = "train"
        out[m_va] = "val"
        return out

    for tf, (tr, va) in cutoffs.items():
        tr64, va64 = _as_dt64(tr), _as_dt64(va)
        m_tf = (tf_col == str(tf).lower().strip())
        if not m_tf.any():
            continue
        idx = np.where(m_tf)[0]
        tt_tf = tt[m_tf]
        m_tr = tt_tf <= tr64
        m_va = (tt_tf > tr64) & (tt_tf <= va64)
        out[idx[m_tr]] = "train"
        out[idx[m_va]] = "val"

    return out
# ----------------------------
# Main split runner per dataset
# ----------------------------
def split_one_dataset(
    fs: pafs.S3FileSystem,
    ds_all: ds.Dataset,
    dataset_name: str,
    dst_stage2_root: str,
    cutoffs: dict,
    symbols: Optional[List[str]],
    tfs: Optional[List[str]],
    batch_size: int,
    write_batch_rows: int,
    eventid_max_keys: int,
):
    schema = ds_all.schema
    cols = list(schema.names)

    # hard requirements new Stage2
    must = ["row_id", "event_id", "task", "t", "tf", "symbol", "year"]
    missing = [c for c in must if c not in cols]
    if missing:
        raise ValueError(f"[{dataset_name}] Stage2(all) missing required columns: {missing}")

    # optional filter (scanner-level)
    expr = None
    if symbols:
        expr = ds.field("symbol").isin([str(s) for s in symbols])
    if tfs:
        e2 = ds.field("tf").isin([str(x).lower().strip() for x in tfs])
        expr = e2 if expr is None else (expr & e2)

    scanner = ds_all.scanner(columns=cols, filter=expr, batch_size=batch_size)

    # Buffers per (split, symbol, year)
    buffers: Dict[str, Dict[Tuple[str, int], List[pd.DataFrame]]] = {"train": defaultdict(list),
                                                                     "val": defaultdict(list),
                                                                     "test": defaultdict(list)}
    buffered_rows: Dict[str, Dict[Tuple[str, int], int]] = {"train": defaultdict(int),
                                                            "val": defaultdict(int),
                                                            "test": defaultdict(int)}

    part_id: Dict[str, Dict[Tuple[str, int], int]] = {"train": defaultdict(int),
                                                      "val": defaultdict(int),
                                                      "test": defaultdict(int)}

    totals = {"train": 0, "val": 0, "test": 0}

    uniq = UniqueEventIdChecks(max_keys=int(eventid_max_keys))
    formula = EventFormulaChecks()
    taskchk = TaskChecks(dataset_name=str(dataset_name))
    
    
    def _dst_path(split: str, sym: str, year: int, part: int) -> str:
        return (
            f"{dst_stage2_root.rstrip('/')}/{split}/{dataset_name}/"
            f"{sym}/{int(year)}/parts/part-{part:05d}.parquet"
        )

    def flush(split: str, key: Tuple[str, int]):
        lst = buffers[split].get(key)
        if not lst:
            return
        out_df = pd.concat(lst, ignore_index=True)
        buffers[split][key].clear()
        buffered_rows[split][key] = 0

        sym, year = key
        p = part_id[split][key]
        part_id[split][key] += 1

        path = _dst_path(split, sym, year, p)
        _write_parquet(fs, path, out_df)
        totals[split] += len(out_df)

        print(f"[write] {dataset_name}/{split} {sym}/{year} part={p:05d} rows={len(out_df):,} total_split={totals[split]:,}")

    for b in scanner.to_batches():
        df = b.to_pandas(types_mapper=pd.ArrowDtype)
        if df.empty:
            continue

        # normalize + filter bad t
        df["t"] = pd.to_datetime(df["t"], utc=True, errors="coerce").dt.tz_convert(None)
        df = df.dropna(subset=["t"])
        if df.empty:
            continue

        # streaming validation
        formula.process(df)
        uniq.process(df["event_id"])
        taskchk.process(df)

        # assign split with shared cutoffs
        split_arr = assign_split(df, cutoffs)
        df["__split"] = split_arr

        # write by (split, symbol, year)
        df["symbol"] = df["symbol"].astype(str)
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")

        for sp in ("train", "val", "test"):
            sub = df[df["__split"] == sp].drop(columns=["__split"])
            if sub.empty:
                continue

            # group within batch to keep partition dirs
            for (sym, yr), g in sub.groupby(["symbol", "year"], sort=False):
                if pd.isna(yr):
                    continue
                key = (str(sym), int(yr))
                buffers[sp][key].append(g)
                buffered_rows[sp][key] += len(g)

                if buffered_rows[sp][key] >= int(write_batch_rows):
                    flush(sp, key)

    # final flush
    for sp in ("train", "val", "test"):
        for key in list(buffers[sp].keys()):
            flush(sp, key)

    # finalize validations
    formula.finalize_or_raise(tag=f"{dataset_name}")
    uniq.finalize_or_raise(tag=f"{dataset_name}")
    taskchk.finalize_or_raise(tag=f"{dataset_name}")

    print(f"[done] {dataset_name} totals: {totals}")
    return totals

# ----------------------------
# CLI
# ----------------------------
def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("--src-stage2-all-root", default="s3://tradebot-config-tokyo/data/stage2/all",
                    help="Root all/: contient all/go/... et all/dir/...")
    # IMPORTANT: recommande de pointer vers s3://.../data/stage2 (pas .../stage2/split)
    ap.add_argument("--dst-stage2-split-root", default="s3://tradebot-config-tokyo/data/stage2/split",
                    help="Root destination: écrit {train,val,test}/{go,dir}/{SYMBOL}/{YEAR}/parts/*.parquet et _meta/split_cutoffs.json")

    ap.add_argument("--aws-region", default="ap-northeast-1")

    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--tfs", nargs="*", default=None)

    ap.add_argument("--q-train", type=float, default=0.70)
    ap.add_argument("--q-val", type=float, default=0.85)

    ap.add_argument("--per-tf", action="store_true", help="cutoffs par tf (recommandé)")
    ap.add_argument("--global", dest="per_tf", action="store_false", help="cutoffs globaux")
    ap.set_defaults(per_tf=True)

    ap.add_argument("--batch-size", type=int, default=400_000)
    ap.add_argument("--write-batch-rows", type=int, default=250_000)

    ap.add_argument("--eventid-max-keys", type=int, default=5_000_000,
                    help="Cap du set(hash(event_id)) par dataset (RAM).")

    ap.add_argument("--fit-on", choices=["go", "dir"], default="go",
                    help="Dataset utilisé pour fitter les cutoffs (recommandé: go).")

    return ap.parse_args()

def main():
    args = parse_args()
    fs = _fs(args.aws_region)

    # 1) Fit cutoffs une seule fois (sur go par défaut)
    ds_fit = dataset_stage2_all(fs, args.src_stage2_all_root, args.fit_on)
    policy = SplitPolicy(q_train_end=args.q_train, q_val_end=args.q_val, per_tf=bool(args.per_tf))

    print(f"[split] fitting cutoffs on dataset={args.fit_on} ...")
    t0 = time.time()
    cutoffs = fit_time_cutoffs(
        ds_fit, policy,
        symbols=args.symbols, tfs=args.tfs,
        batch_size=args.batch_size,
    )
    save_cutoffs_json(fs, args.dst_stage2_split_root, cutoffs)
    print(f"[split] saved shared cutoffs to {args.dst_stage2_split_root.rstrip('/')}/_meta/split_cutoffs.json dt={time.time()-t0:.1f}s")

    # 2) Split GO + DIR avec les mêmes cutoffs
    totals = {}
    for dname in ("go", "dir"):
        print("=" * 80)
        print(f"[run] splitting dataset={dname}")
        print("=" * 80)

        ds_all = dataset_stage2_all(fs, args.src_stage2_all_root, dname)

        totals[dname] = split_one_dataset(
            fs=fs,
            ds_all=ds_all,
            dataset_name=dname,
            dst_stage2_root=args.dst_stage2_split_root,
            cutoffs=cutoffs,
            symbols=args.symbols,
            tfs=args.tfs,
            batch_size=args.batch_size,
            write_batch_rows=args.write_batch_rows,
            eventid_max_keys=int(args.eventid_max_keys),
        )

    print("[done] totals:", totals)
    print("[done] shared cutoffs used for both go and dir: OK")

if __name__ == "__main__":
    main()