#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
validate_stage2_min.py — "minimum vital" Stage2 (post-split)

Objectif:
- Vérifier que Stage2 a produit un dataset exploitable pour Stage3/Stage4
- Fail fast uniquement sur: colonnes critiques, labels, audits flags, event_id formula, task/dataset, TF non vide
- Memory-safe: lecture par batches, pas de "schema consistency" globale, unicité event_id optionnelle et capée

Layouts supportés (auto):
A) <root>/<split>/<dataset>/<sym>/<year>/parts/part-*.parquet
B) <root>/<dataset>/<split>/<sym>/<year>/parts/part-*.parquet

Ex: s3://tradebot-config-tokyo/data/stage2/split/go/train/BTCUSDT/2024/parts/part-*.parquet
"""

from __future__ import annotations

import argparse
import re
import gc
import hashlib
from dataclasses import dataclass, field
from collections import Counter, defaultdict
from typing import Optional, List, Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import fsspec
from botocore.exceptions import ClientError


# ----------------------------
# Helpers
# ----------------------------
def log(msg: str) -> None:
    print(msg, flush=True)

def so_s3(region: Optional[str], anon: bool, requester_pays: bool) -> dict:
    so: Dict[str, object] = {}
    if region:
        so["client_kwargs"] = {"region_name": region}
    if anon:
        so["anon"] = True
    if requester_pays:
        so["requester_pays"] = True
    return so

def open_parquet(path: str, so: dict) -> pq.ParquetFile:
    fo = fsspec.open(path, "rb", **(so or {})).open()
    return pq.ParquetFile(fo)

def ensure_proto(fs, p: str) -> str:
    proto = fs.protocol[0] if isinstance(fs.protocol, (list, tuple)) else fs.protocol
    if not re.match(r"^[a-z0-9]+://", str(p)):
        return f"{proto}://{p}"
    return str(p)

def glob_one(pat: str, so: dict, debug: bool = False) -> List[str]:
    if debug:
        log(f"[glob] {pat}")
    try:
        fs, _, paths = fsspec.get_fs_token_paths(pat, storage_options=so)
        out = [ensure_proto(fs, p) for p in sorted(paths)]
        if debug:
            log(f"[glob] -> {len(out)}")
        return out
    except PermissionError:
        log(f"❌ AccessDenied glob: {pat}")
        return []
    except ClientError as e:
        log(f"❌ ClientError glob: {pat} — {e}")
        return []
    except Exception as e:
        log(f"❌ glob error: {pat} — {e!r}")
        return []

def norm_tf(x) -> str:
    return str(x).strip().lower()

def norm_task(x) -> str:
    return str(x).strip().lower()

NA_STRINGS = {"", "na", "n/a", "nan", "none", "null"}

def is_na_text(s: pd.Series) -> pd.Series:
    x = s.astype(str).str.strip().str.lower()
    return x.isin(NA_STRINGS)

def hash64(s: str) -> int:
    h = hashlib.blake2b(s.encode("utf-8", "ignore"), digest_size=8).digest()
    return int.from_bytes(h, "big", signed=False)


# ----------------------------
# Contract (minimum vital)
# ----------------------------
REQUIRED_STRUCTURE = ["t", "symbol", "year", "tf", "row_id", "task", "event_id"]
REQUIRED_LABELS = ["y_go", "y_dir"]

# Stage2 doit porter le budget gate (déjà required)
REQUIRED_EV = ["audit_p_thr_ev0"]

# Flags audit (déjà required)
REQUIRED_AUDIT_FLAGS = ["audit_market_toxic", "audit_timeout", "audit_early_abort"]

# ----------------------------
# REQUIRED audit numeric (Stage2 core economics)
# ----------------------------
REQUIRED_AUDIT_NUM = [
    "audit_pnl_net_bps",
    "audit_tp_bps",
    "audit_sl_bps",
    "audit_rr_min",
    "audit_risk_r_bps",
]

# ----------------------------
# REQUIRED audit text / enums
# ----------------------------
REQUIRED_AUDIT_TEXT = [
    "audit_exit_reason",
    "audit_fill_mode",
]

# --- NEW: required audit extras (oracle-safe, non-features mais audits) ---
REQUIRED_AUDIT_EXTRA_NUM = [
    # MFE/MAE in R
    "audit_mfe_R_L", "audit_mae_R_L",
    "audit_mfe_R_S", "audit_mae_R_S",

    # first touch + step
    "audit_first_touch_L", "audit_first_touch_step_L",
    "audit_first_touch_S", "audit_first_touch_step_S",
]

REQUIRED_AUDIT_EXTRA_HIT_FLAGS = [
    # hit +kR
    "audit_hit_p1R_L","audit_hit_p2R_L","audit_hit_p3R_L","audit_hit_p4R_L","audit_hit_p5R_L",
    "audit_hit_p1R_S","audit_hit_p2R_S","audit_hit_p3R_S","audit_hit_p4R_S","audit_hit_p5R_S",

    # hit -kR
    "audit_hit_m1R_L","audit_hit_m2R_L","audit_hit_m3R_L",
    "audit_hit_m1R_S","audit_hit_m2R_S","audit_hit_m3R_S",
]

# Colonnes oracle/leakage: on garde le strict sur Y/y et regex,
# mais sup_* devient WARNING (voir plus bas)
FORBIDDEN_EXACT = {"Y", "y"}
FORBIDDEN_REGEX = [r"^Y_", r"^label_"]

# WARNING only:
WARNING_PREFIXES = ("sup_",)

# ----------------------------
# Stats streaming
# ----------------------------
@dataclass
class Stats:
    rows: int = 0
    tf_counts: Counter = field(default_factory=Counter)
    task_counts: Counter = field(default_factory=Counter)
    ygo_counts: Counter = field(default_factory=Counter)
    ydir_counts: Counter = field(default_factory=Counter)
    nulls: Counter = field(default_factory=Counter)
    sup_cols_seen: Counter = field(default_factory=Counter)

    ok_task: int = 0
    tot_task: int = 0

    ok_tf_norm: int = 0
    tot_tf_norm: int = 0

    ok_event_formula: int = 0
    tot_event_formula: int = 0

    ok_ygo: int = 0
    tot_ygo: int = 0

    ok_ydir: int = 0
    tot_ydir: int = 0

    ok_audit_flags: int = 0
    tot_audit_flags: int = 0

    ok_audit_logic: int = 0
    tot_audit_logic: int = 0

    # optionally caped uniqueness check
    seen: set = field(default_factory=set)
    seen_cap: int = 2_000_000
    dup_eid: int = 0
    null_eid: int = 0
    uniqueness_enabled: bool = True


def fmt_ratio(ok: int, tot: int) -> str:
    return "NA" if tot == 0 else f"{ok}/{tot} = {ok/tot:.2%}"


# ----------------------------
# Discovery: support 2 layouts
# ----------------------------
def discover_paths(root: str, dataset: str, split: str,
                   symbols: Optional[List[str]], years: Optional[List[int]],
                   so: dict, debug: bool = False) -> List[str]:
    root = root.rstrip("/")
    syms = symbols or ["*"]
    yrs = [str(y) for y in (years or ["*"])]

    pats = []
    # Layout A: <root>/<split>/<dataset>/...
    pats += [f"{root}/{split}/{dataset}/{s}/{y}/parts/part-*.parquet" for s in syms for y in yrs]
    # Layout B: <root>/<dataset>/<split>/...
    pats += [f"{root}/{dataset}/{split}/{s}/{y}/parts/part-*.parquet" for s in syms for y in yrs]

    out: List[str] = []
    for pat in pats:
        out.extend(glob_one(pat, so, debug=debug))

    out = sorted(set(out))
    return out


# ----------------------------
# Validation core
# ----------------------------
def validate_files(paths: List[str], dataset: str, so: dict,
                   batch_rows: int, stats: Stats,
                   symbols: Optional[List[str]], years: Optional[List[int]]) -> None:
    if not paths:
        log("⚠️ Aucun fichier trouvé.")
        return

    log(f"✅ {len(paths)} fichier(s) (top 10):")
    for p in paths[:10]:
        log(f"  - {p}")
    if len(paths) > 10:
        log(f"  ... (+{len(paths)-10})")

    dataset = dataset.lower().strip()
    if dataset not in {"go", "dir"}:
        raise ValueError("dataset must be go|dir")

    symset = set(map(str, symbols)) if symbols else None
    yrset = set(int(y) for y in years) if years else None

    # Helpers for side-aware requirements
    def _exit_norm(s: pd.Series) -> pd.Series:
        return s.astype(str).str.strip().str.upper()

    def _is_none_like_exit(v: pd.Series) -> pd.Series:
        vv = _exit_norm(v)
        return vv.isin(["NONE", "NOFILL", ""])

    def _need_L(v: pd.Series) -> pd.Series:
        vv = _exit_norm(v)
        return vv.str.contains("_LONG", regex=False)

    def _need_S(v: pd.Series) -> pd.Series:
        vv = _exit_norm(v)
        return vv.str.contains("_SHORT", regex=False)

    def _is_time(v: pd.Series) -> pd.Series:
        return _exit_norm(v).eq("TIME")

    # Map side -> required columns
    REQ_NUM_L = ["audit_mfe_R_L", "audit_mae_R_L", "audit_first_touch_step_L"]
    REQ_NUM_S = ["audit_mfe_R_S", "audit_mae_R_S", "audit_first_touch_step_S"]

    # first_touch is NUMERIC in your Stage2 (observed -1)
    REQ_TOUCH_L = ["audit_first_touch_L"]
    REQ_TOUCH_S = ["audit_first_touch_S"]
    VALID_TOUCH_INT = {-1, 0, 1, 2}  # permissif : -1 = none/unknown, 1/2 = tp/sl (selon convention)

    REQ_HIT_L = [c for c in REQUIRED_AUDIT_EXTRA_HIT_FLAGS if c.endswith("_L")]
    REQ_HIT_S = [c for c in REQUIRED_AUDIT_EXTRA_HIT_FLAGS if c.endswith("_S")]

    for path in paths:
        pf = open_parquet(path, so)
        cols = list(pf.schema.names)

        # forbid (schema-level)
        bad_exact = [c for c in cols if c in FORBIDDEN_EXACT]
        bad_re = [c for c in cols if any(re.match(rx, str(c)) for rx in FORBIDDEN_REGEX)]

        # WARNING: sup_* only
        warn_pref = [c for c in cols if any(str(c).startswith(px) for px in WARNING_PREFIXES)]
        if warn_pref:
            stats.sup_cols_seen.update(warn_pref)

        if bad_exact or bad_re:
            raise ValueError(
                f"[FORBIDDEN] {path}\n"
                f"  exact={bad_exact}\n"
                f"  re={bad_re}\n"
                f"  warn_sup_prefix={warn_pref[:20]}"
            )

        # required columns exist (schema-level)
        required_all = (
            REQUIRED_STRUCTURE
            + REQUIRED_LABELS
            + REQUIRED_EV
            + REQUIRED_AUDIT_FLAGS
            + REQUIRED_AUDIT_NUM
            + REQUIRED_AUDIT_TEXT
            + REQUIRED_AUDIT_EXTRA_NUM
            + REQUIRED_AUDIT_EXTRA_HIT_FLAGS
        )
        missing = [c for c in required_all if c not in cols]
        if missing:
            raise ValueError(f"[MISSING COLS] {path}: {missing}")

        # read only columns needed for checks
        wanted = sorted(set(required_all))
        wanted = [c for c in wanted if c in cols]

        for batch in pf.iter_batches(batch_size=batch_rows, columns=wanted):
            t = pa.Table.from_batches([batch])
            df = t.to_pandas()

            # optional filtering
            if symset is not None and "symbol" in df.columns:
                df = df[df["symbol"].astype(str).isin(symset)]
            if yrset is not None and "year" in df.columns:
                yy = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
                df = df[yy.isin(yrset)]

            if df.empty:
                del df, t, batch
                gc.collect()
                continue

            n = len(df)
            stats.rows += n

            # null counters
            nn = df.isna().sum().to_dict()
            for k, v in nn.items():
                stats.nulls[k] += int(v)

            # task == dataset
            stats.tot_task += n
            taskn = df["task"].map(norm_task)
            ok_task_mask = (taskn == dataset)
            stats.ok_task += int(ok_task_mask.sum())
            if not ok_task_mask.all():
                ex = df.loc[~ok_task_mask, ["symbol","year","tf","row_id","task","event_id"]].head(5)
                raise ValueError(f"[BAD task] expected '{dataset}'\nfile={path}\n{ex}")

            stats.task_counts.update(taskn.value_counts(dropna=False).to_dict())

            # tf normalized
            tf_raw = df["tf"].astype(str)
            tf_norm = df["tf"].map(norm_tf)
            stats.tot_tf_norm += n
            ok_tf = (tf_norm != "") & (tf_norm != "nan") & (tf_raw == tf_norm)
            stats.ok_tf_norm += int(ok_tf.sum())
            if not ok_tf.all():
                ex = df.loc[~ok_tf, ["symbol","year","row_id","tf","event_id"]].head(5)
                raise ValueError(f"[BAD tf] must be non-empty + already normalized lower/strip\nfile={path}\n{ex}")

            stats.tf_counts.update(tf_norm.value_counts(dropna=False).to_dict())

            # labels domain
            ygo = pd.to_numeric(df["y_go"], errors="coerce")
            m = ygo.notna()
            stats.tot_ygo += int(m.sum())
            stats.ok_ygo += int(((ygo[m] == 0) | (ygo[m] == 1)).sum())
            if stats.ok_ygo != stats.tot_ygo:
                ex = df.loc[m & ~((ygo == 0) | (ygo == 1)), ["symbol","year","tf","row_id","y_go","event_id"]].head(5)
                raise ValueError(f"[BAD y_go] must be in {{0,1}}\nfile={path}\n{ex}")
            stats.ygo_counts.update(df["y_go"].value_counts(dropna=False).to_dict())

            ydir = pd.to_numeric(df["y_dir"], errors="coerce")
            m = ydir.notna()
            stats.tot_ydir += int(m.sum())
            if dataset == "dir":
                ok = (ydir[m] == -1) | (ydir[m] == 1)
            else:
                ok = (ydir[m] == -1) | (ydir[m] == 0) | (ydir[m] == 1)
            stats.ok_ydir += int(ok.sum())
            if stats.ok_ydir != stats.tot_ydir:
                ex = df.loc[m & ~ok, ["symbol","year","tf","row_id","y_dir","event_id"]].head(5)
                raise ValueError(f"[BAD y_dir] domain mismatch for dataset={dataset}\nfile={path}\n{ex}")
            stats.ydir_counts.update(df["y_dir"].value_counts(dropna=False).to_dict())

            # required audit text
            v = df["audit_exit_reason"].astype(str).str.strip().str.upper()
            bad = v.isin(["", "NAN", "NULL"])   # <-- NONE est autorisé
            if bad.any():
                ex = df.loc[bad, ["symbol","year","tf","row_id","audit_exit_reason","event_id"]].head(5)
                raise ValueError(f"[BAD audit_exit_reason] empty/null\nfile={path}\n{ex}")

            v = df["audit_fill_mode"].astype(str).str.strip().str.lower()
            bad = v.isin(["", "nan", "none", "null"])
            if bad.any():
                ex = df.loc[bad, ["symbol","year","tf","row_id","audit_fill_mode","event_id"]].head(5)
                raise ValueError(f"[BAD audit_fill_mode] empty/null\nfile={path}\n{ex}")

            # required audit numeric (core): non-null + finite
            for c in REQUIRED_AUDIT_NUM:
                s = pd.to_numeric(df[c], errors="coerce").astype("float64")
                if s.isna().any():
                    ex = df.loc[s.isna(), ["symbol","year","tf","row_id",c,"event_id"]].head(5)
                    raise ValueError(f"[BAD required audit num] {c} contains NaN/NULL\nfile={path}\n{ex}")
                if not np.isfinite(s.to_numpy()).all():
                    bad = ~np.isfinite(s.to_numpy())
                    ex = df.loc[bad, ["symbol","year","tf","row_id",c,"event_id"]].head(5)
                    raise ValueError(f"[BAD required audit num] {c} contains inf/NaN\nfile={path}\n{ex}")

            # audit_p_thr_ev0 checks
            pthr = pd.to_numeric(df["audit_p_thr_ev0"], errors="coerce")
            m_thr = pthr.notna()
            if not m_thr.any():
                ex = df.loc[:, ["symbol","year","tf","row_id","event_id"]].head(5)
                raise ValueError(f"[BAD audit_p_thr_ev0] all values are null/NaN\nfile={path}\n{ex}")

            bad_finite = m_thr & ~np.isfinite(pthr.to_numpy(dtype="float64"))
            if bad_finite.any():
                ex = df.loc[bad_finite, ["symbol","year","tf","row_id","audit_p_thr_ev0","event_id"]].head(5)
                raise ValueError(f"[BAD audit_p_thr_ev0] contains inf/NaN\nfile={path}\n{ex}")

            bad_range = (pthr[m_thr] < -1e-6) | (pthr[m_thr] > 1.0 + 1e-6)
            if bad_range.any():
                ex = df.loc[m_thr & bad_range, ["symbol","year","tf","row_id","audit_p_thr_ev0","event_id"]].head(5)
                raise ValueError(f"[BAD audit_p_thr_ev0] out of [0,1]\nfile={path}\n{ex}")

            # event_id formula row_id|task|tf
            eid = df["event_id"]
            rid = df["row_id"].astype(str)
            want = rid + "|" + taskn.astype(str) + "|" + tf_norm.astype(str)
            good_mask = eid.notna() & df["row_id"].notna() & df["task"].notna() & df["tf"].notna()
            stats.tot_event_formula += int(good_mask.sum())
            if good_mask.any():
                okf = (eid[good_mask].astype(str).values == want[good_mask].values)
                stats.ok_event_formula += int(okf.sum())
                if not okf.all():
                    badm = good_mask.copy()
                    badm.loc[good_mask] = ~okf
                    ex = df.loc[badm, ["symbol","year","tf","row_id","task","event_id"]].head(5)
                    raise ValueError(f"[BAD event_id] expected row_id|task|tf\nfile={path}\n{ex}")

            # uniqueness (caped)
            if stats.uniqueness_enabled:
                is_null = eid.isna()
                stats.null_eid += int(is_null.sum())
                for s in eid[~is_null].astype(str).values:
                    h = hash64(s)
                    if h in stats.seen:
                        stats.dup_eid += 1
                    else:
                        if len(stats.seen) < stats.seen_cap:
                            stats.seen.add(h)

            # audit flags 0/1
            for c in REQUIRED_AUDIT_FLAGS:
                s = pd.to_numeric(df[c], errors="coerce")
                m = s.notna()
                stats.tot_audit_flags += int(m.sum())
                stats.ok_audit_flags += int(((s[m] == 0) | (s[m] == 1)).sum())
                if (m & ~((s == 0) | (s == 1))).any():
                    ex = df.loc[m & ~((s == 0) | (s == 1)), ["symbol","year","tf","row_id",c,"event_id"]].head(5)
                    raise ValueError(f"[BAD audit flag] {c} must be in {{0,1}}\nfile={path}\n{ex}")

            # minimal audit logic: if exit_reason is set => pnl notna
            exit_na = is_na_text(df["audit_exit_reason"])
            pnl = pd.to_numeric(df["audit_pnl_net_bps"], errors="coerce")
            stats.tot_audit_logic += n
            ok_logic = exit_na | pnl.notna()
            stats.ok_audit_logic += int(ok_logic.sum())
            if not ok_logic.all():
                ex = df.loc[~ok_logic, ["symbol","year","tf","row_id","audit_exit_reason","audit_pnl_net_bps","event_id"]].head(5)
                raise ValueError(f"[BAD audit logic] pnl missing while exit_reason set\nfile={path}\n{ex}")

            # -------------------------------------------------------
            # SIDE-AWARE audit extras checks
            # -------------------------------------------------------
            exit_reason = df["audit_exit_reason"]
            none_like = _is_none_like_exit(exit_reason)

            # audit "complete": exit not NONE/NOFILL and flags==0
            f_early = pd.to_numeric(df["audit_early_abort"], errors="coerce").fillna(0).astype("int64")
            f_timeo = pd.to_numeric(df["audit_timeout"], errors="coerce").fillna(0).astype("int64")
            f_tox   = pd.to_numeric(df["audit_market_toxic"], errors="coerce").fillna(0).astype("int64")
            flags_ok = (f_early == 0) & (f_timeo == 0) & (f_tox == 0)

            complete = (~none_like) & flags_ok

            needL = complete & _need_L(exit_reason)
            needS = complete & _need_S(exit_reason)
            isTIME = complete & _is_time(exit_reason)

            # TIME: allow "at least one side has mfe+mae notna"
            if isTIME.any():
                mfeL = pd.to_numeric(df["audit_mfe_R_L"], errors="coerce")
                maeL = pd.to_numeric(df["audit_mae_R_L"], errors="coerce")
                mfeS = pd.to_numeric(df["audit_mfe_R_S"], errors="coerce")
                maeS = pd.to_numeric(df["audit_mae_R_S"], errors="coerce")
                hasL = mfeL.notna() & maeL.notna()
                hasS = mfeS.notna() & maeS.notna()
                bad = isTIME & ~(hasL | hasS)
                if bad.any():
                    ex = df.loc[bad, ["symbol","year","tf","row_id","event_id","audit_exit_reason",
                                      "audit_mfe_R_L","audit_mae_R_L","audit_mfe_R_S","audit_mae_R_S"]].head(5)
                    raise ValueError(f"[BAD audit extras] TIME requires at least one side (L or S) to have mfe+mae\nfile={path}\n{ex}")

            # Require L-side extras only when exit is *_LONG
            if needL.any():
                # numeric (mfe/mae finite; step allows -1)
                for c in ["audit_mfe_R_L", "audit_mae_R_L"]:
                    s = pd.to_numeric(df[c], errors="coerce").astype("float64")
                    bad = needL & (s.isna() | ~np.isfinite(s.to_numpy()))
                    if bad.any():
                        ex = df.loc[bad, ["symbol","year","tf","row_id","event_id","audit_exit_reason",c]].head(5)
                        raise ValueError(f"[BAD required audit num] {c} missing/inf for *_LONG\nfile={path}\n{ex}")

                s = pd.to_numeric(df["audit_first_touch_step_L"], errors="coerce").astype("float64")
                bad = needL & (s.isna() | ~np.isfinite(s.to_numpy()))
                if bad.any():
                    ex = df.loc[bad, ["symbol","year","tf","row_id","event_id","audit_exit_reason","audit_first_touch_step_L"]].head(5)
                    raise ValueError(f"[BAD required audit num] audit_first_touch_step_L missing/inf for *_LONG\nfile={path}\n{ex}")
                # allow -1 (unknown/no-touch) for now

                # first_touch numeric domain
                ft = pd.to_numeric(df["audit_first_touch_L"], errors="coerce")
                bad = needL & (ft.isna() | ~ft.isin(list(VALID_TOUCH_INT)))
                if bad.any():
                    ex = df.loc[bad, ["symbol","year","tf","row_id","event_id","audit_exit_reason","audit_first_touch_L"]].head(5)
                    raise ValueError(f"[BAD required audit code] audit_first_touch_L invalid for *_LONG (want {sorted(VALID_TOUCH_INT)})\nfile={path}\n{ex}")

                # hit flags
                for c in REQ_HIT_L:
                    s = pd.to_numeric(df[c], errors="coerce")
                    bad = needL & (s.isna() | ~((s == 0) | (s == 1)))
                    if bad.any():
                        ex = df.loc[bad, ["symbol","year","tf","row_id","event_id","audit_exit_reason",c]].head(5)
                        raise ValueError(f"[BAD required audit hit flag] {c} invalid/missing for *_LONG\nfile={path}\n{ex}")

            # Require S-side extras only when exit is *_SHORT
            if needS.any():
                for c in ["audit_mfe_R_S", "audit_mae_R_S"]:
                    s = pd.to_numeric(df[c], errors="coerce").astype("float64")
                    bad = needS & (s.isna() | ~np.isfinite(s.to_numpy()))
                    if bad.any():
                        ex = df.loc[bad, ["symbol","year","tf","row_id","event_id","audit_exit_reason",c]].head(5)
                        raise ValueError(f"[BAD required audit num] {c} missing/inf for *_SHORT\nfile={path}\n{ex}")

                s = pd.to_numeric(df["audit_first_touch_step_S"], errors="coerce").astype("float64")
                bad = needS & (s.isna() | ~np.isfinite(s.to_numpy()))
                if bad.any():
                    ex = df.loc[bad, ["symbol","year","tf","row_id","event_id","audit_exit_reason","audit_first_touch_step_S"]].head(5)
                    raise ValueError(f"[BAD required audit num] audit_first_touch_step_S missing/inf for *_SHORT\nfile={path}\n{ex}")

                ft = pd.to_numeric(df["audit_first_touch_S"], errors="coerce")
                bad = needS & (ft.isna() | ~ft.isin(list(VALID_TOUCH_INT)))
                if bad.any():
                    ex = df.loc[bad, ["symbol","year","tf","row_id","event_id","audit_exit_reason","audit_first_touch_S"]].head(5)
                    raise ValueError(f"[BAD required audit code] audit_first_touch_S invalid for *_SHORT (want {sorted(VALID_TOUCH_INT)})\nfile={path}\n{ex}")

                for c in REQ_HIT_S:
                    s = pd.to_numeric(df[c], errors="coerce")
                    bad = needS & (s.isna() | ~((s == 0) | (s == 1)))
                    if bad.any():
                        ex = df.loc[bad, ["symbol","year","tf","row_id","event_id","audit_exit_reason",c]].head(5)
                        raise ValueError(f"[BAD required audit hit flag] {c} invalid/missing for *_SHORT\nfile={path}\n{ex}")

            del df, t, batch
            gc.collect()

def print_summary(stats: Stats, dataset: str, split: str) -> None:
    log("\n" + "="*80)
    log(f"SUMMARY dataset={dataset} split={split} rows={stats.rows:,}")
    log("="*80)

    log(f"task==dataset: {fmt_ratio(stats.ok_task, stats.tot_task)}")
    log(f"tf normalized+nonempty: {fmt_ratio(stats.ok_tf_norm, stats.tot_tf_norm)}")
    log(f"event_id formula ok: {fmt_ratio(stats.ok_event_formula, stats.tot_event_formula)}")
    log(f"y_go domain ok: {fmt_ratio(stats.ok_ygo, stats.tot_ygo)}")
    log(f"y_dir domain ok: {fmt_ratio(stats.ok_ydir, stats.tot_ydir)}")
    log(f"audit flags domain ok: {fmt_ratio(stats.ok_audit_flags, stats.tot_audit_flags)}")
    if stats.tot_audit_logic:
        log(f"audit logic ok: {fmt_ratio(stats.ok_audit_logic, stats.tot_audit_logic)}")

    log("\nTF counts (top):")
    for k, v in stats.tf_counts.most_common(10):
        log(f"  {k}: {v:,}")

    log("\ny_go counts:")
    for k, v in stats.ygo_counts.most_common():
        log(f"  {k}: {v:,}")

    log("\ny_dir counts:")
    for k, v in stats.ydir_counts.most_common():
        log(f"  {k}: {v:,}")

    log("\nNull rates (top 15):")
    for c, n in stats.nulls.most_common(15):
        log(f"  {c:28s} {n:>10,d}  ({n/max(stats.rows,1):.2%})")

    if stats.uniqueness_enabled:
        if len(stats.seen) >= stats.seen_cap:
            log(f"\n[WARN] event_id uniqueness: cap reached ({stats.seen_cap:,}). dup_eid reported is a LOWER bound.")
        log(f"event_id null: {stats.null_eid:,} | dup_eid (hashed): {stats.dup_eid:,}")
    
    if stats.sup_cols_seen:
        log("\n[WARN] sup_* columns detected (schema-level):")
        for c, n in stats.sup_cols_seen.most_common(30):
            log(f"  {c}: {n} file(s)")
        if len(stats.sup_cols_seen) > 30:
            log(f"  ... (+{len(stats.sup_cols_seen)-30})")

    log("\n✅ Validation OK.\n")


# ----------------------------
# CLI
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser("validate_stage2_min (minimum vital)")
    p.add_argument("--src-root", default="s3://tradebot-config-tokyo/data/stage2",
                   help="Root Stage2. Supporte layouts A/B automatiquement.")
    p.add_argument("--dataset", choices=["go", "dir"], required=True)
    p.add_argument("--split", choices=["train", "val", "test", "all", "tvt"], default="tvt")
    p.add_argument("--symbols", nargs="*", default=None)
    p.add_argument("--years", nargs="*", type=int, default=None)

    p.add_argument("--s3-region", default="ap-northeast-1")
    p.add_argument("--s3-anon", action="store_true")
    p.add_argument("--s3-requester-pays", action="store_true")

    p.add_argument("--batch-rows", type=int, default=200_000)
    p.add_argument("--debug", action="store_true")

    p.add_argument("--no-uniq", action="store_true",
                   help="Désactive la vérif d'unicité event_id (hash-set capé).")
    p.add_argument("--uniq-cap", type=int, default=2_000_000,
                   help="Cap du set d'event_id hashés.")
    return p.parse_args()


def main():
    args = parse_args()
    so = so_s3(args.s3_region, args.s3_anon, args.s3_requester_pays)

    if args.split == "tvt":
        splits = ["train", "val", "test"]
    else:
        splits = [args.split]

    for sp in splits:
        log("\n" + "="*80)
        log(f"🔎 dataset={args.dataset} split={sp}")
        log("="*80)

        paths = discover_paths(args.src_root, args.dataset, sp, args.symbols, args.years, so, debug=args.debug)
        if not paths:
            log("⚠️ aucun fichier")
            continue

        stats = Stats()
        stats.uniqueness_enabled = (not args.no_uniq)
        stats.seen_cap = int(args.uniq_cap)

        validate_files(
            paths=paths,
            dataset=args.dataset,
            so=so,
            batch_rows=args.batch_rows,
            stats=stats,
            symbols=args.symbols,
            years=args.years,
        )

        print_summary(stats, dataset=args.dataset, split=sp)


if __name__ == "__main__":
    main()