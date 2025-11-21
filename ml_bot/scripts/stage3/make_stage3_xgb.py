#!/usr/bin/env python3
# make_stage3_xgb.py — Consomme Stage2 {train,val,test} ➜ CSV normalisés (XGBoost)
from __future__ import annotations
import argparse, io, json, sys
from typing import Optional, Dict, Tuple, List
from collections import defaultdict
from uuid import uuid4
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.fs as pafs
import pyarrow.parquet as pq

# =======================
# Feature policy
# =======================

WINSOR_FEATURES = ["cum_depth_within_5bps_opp"]  # extensible
WINSOR_Q_LOW = 0.01
WINSOR_Q_HIGH = 0.99

# seuil "quasi constant" (std très faible) pour le masque _isconstant
CONST_STD_THRESH = 1e-6  # safe (zéro bruit). Tu peux tester 5e-4 si tu veux plus agressif.

# flags "mur" (seuils simples)
WALL_THRESHOLDS = {
    "wall_opp_share_5":  [0.08],     # ex: un seul seuil
    "wall_opp_share_15": [0.05, 0.12],
}

META_GG_STATS = "_meta/gg_stats.json"

LOG1P_FEATURES = [
    "entry",
    "cum_depth_within_5bps_opp",
    "cum_depth_within_10bps_opp",
]

ZSCORE_FEATURES = [
    "spread_bps_entry",
    "quote_churn_10s",
    "wall_opp_share_5",
    "wall_opp_share_15",
    "obi_5", "obi_15",
    "microprice_bias",
    "slope_bid_5", "slope_ask_5",
    "slope_bid_15", "slope_ask_15",
    "aggr_ratio_10s",
    "net_delta_15s",
    "ret_stdev_1s_10s_bps",
    "mid_jump_bps_3s",
    "bb_width",
    "adx",
    "atr_percentile",
    "atr_bps",
    "executed_vs_added_ratio",
    "lf_bid_absorb_ratio_3s",
    "lf_bid_refill_ticks_3s",
    # "lf_mid_minus_vwap_3s_bps",   <-- RETIRÉ TEMPORAIREMENT
    "lf_bb_width_pct",
    "lf_atr_rank_30m",
    "lf_ask_wall_decay_3s",

    # --- ajouter pour v47 ---
    "bt_dom_3s",
    "bt_dom_5s",
    "bt_dom_10s",
    "bt_dom_3s_side",
    "bt_dom_5s_side",
    "bt_dom_10s_side",
    "pullback_pos_vs_swing_low",
    "pullback_pos_vs_swing_low_side",

    # versions *_side
    "obi_5_side", "obi_15_side",
    "microprice_bias_side",
    "slope_bid_5_side", "slope_ask_5_side",
    "slope_bid_15_side", "slope_ask_15_side",
    "aggr_ratio_10s_side",
    "net_delta_15s_side",
    "mid_jump_bps_3s_side",
]

LATENCY_FEATURES = ["exchange_latency_ms"]

ALWAYS_KEEP = ["t", "symbol", "tf", "side", "Y"]

# Colonnes à lire pour calculer des poids, mais jamais utilisées comme features (anti-fuite)
WEIGHT_ONLY_COLS = {"THRESH_BPS", "pnl_net_max_bps"}

# Toute colonne listée ici sera drop avant feature encoding (anti-fuite)
EXCLUDE_COLS = {"THRESH_BPS"}  # pnl_net_max_bps n'est pas dans les features de toute façon

# === Robust normalization (par (symbol, TF)) ===
ROBUST_EPS = 1e-9
ROBUST_SCALE_FLOOR = 0.25  # plancher d’échelle pour éviter des std déraisonnables par groupe

# === Stage3 scaling config ==========================================
MICROPRICE_FIX_MODE = "asinh_then_z"   # "pooled_z" | "asinh_then_z"
ROBUST_CLIP_Q = (0.005, 0.995)     # resserré vs 0.001/0.999

HEAVY_TAIL_COLS = {
    "cum_depth_within_5bps_opp",
    "cum_depth_within_10bps_opp",
    "ret_stdev_1s_10s_bps",
    "quote_churn_10s",
    "executed_vs_added_ratio",
    # optionnel si besoin :
    # "spread_bps_entry",
}

# ====================================================================

# Imputation neutre + masques
IMPUTE_NEUTRAL = {
    "aggr_ratio_10s": 0.5,
    "aggr_ratio_15s": 0.5,
    "net_delta_15s":  0.0,
    "aggr_ratio_10s_side": 0.0,
    "aggr_ratio_15s_side": 0.0,
    "net_delta_15s_side":  0.0,
}

# --- PATCH: features signées avec queues lourdes → asinh (log symétrique)
ASINH_FEATURES = [
    "slope_bid_5","slope_ask_5","slope_bid_15","slope_ask_15",
    "slope_bid_5_side","slope_ask_5_side","slope_bid_15_side","slope_ask_15_side",
    "mid_jump_bps_3s","mid_jump_bps_3s_side",
    # optionnel si trop skew chez toi :
    # "microprice_bias","microprice_bias_side",
]

def _apply_asinh_inplace(df: pd.DataFrame):
    for f in ASINH_FEATURES:
        if f in df.columns:
            v = pd.to_numeric(df[f], errors="coerce").astype("float64")
            df[f] = np.arcsinh(v)

# =======================
# S3 / helpers
# =======================

def _tf_key(v) -> str:
    return _norm_str(v)  # <<< simplify; one canonical path

def _norm_str(x) -> str:
    if isinstance(x, (bytes, bytearray)):
        return x.decode("utf-8", "ignore").strip()
    x = str(x)
    if x.startswith("b'") and x.endswith("'"):
        x = x[2:-1]
    if x.startswith('b"') and x.endswith('"'):
        x = x[2:-1]
    return x.strip()  # <<< add this

def _normalize_tf_col_inplace(df: pd.DataFrame):
    if "tf" in df.columns:
        df["tf"] = df["tf"].map(_norm_str)

def _append_meta_parquet(fs: pafs.S3FileSystem, base_meta_root: str, split: str, df_meta: pd.DataFrame):
    """
    Écrit un petit parquet par batch:
    s3://.../_meta/splits_parquet/{split}/part-<uuid>.parquet
    """
    if df_meta.empty:
        return
    table = pa.Table.from_pandas(df_meta, preserve_index=False)
    path = f"{base_meta_root.rstrip('/')}/{split}/part-{uuid4().hex}.parquet"
    with fs.open_output_stream(_strip_s3(path)) as out:
        pq.write_table(table, out)

def _save_group_stats(stats, fs, root_dir):
    """Sauvegarde gg_stats avec clés JSON-safe (déjà str)."""
    path = f"{root_dir.rstrip('/')}/{META_GG_STATS}"
    with fs.open_output_stream(_strip_s3(path)) as out:
        out.write(json.dumps(stats, ensure_ascii=False).encode("utf-8"))

def _apply_winsor_and_constant_mask(df: pd.DataFrame,
                                    feature: str,
                                    stats_for_feature: dict,
                                    global_low: float | None = None,
                                    global_high: float | None = None):
    """
    Applique le winsor par groupe et crée <feature>_isconstant si std<seuil.
    `stats_for_feature` peut avoir des clés "(sym_id:tf_id)" (str) ou (sym_id, tf_id) (tuple).
    """
    keys = df["tf_id"].tolist()

    def _lookup(key_tuple, field):
        # accepte tuple (si present) ou clé sérialisée "si:ti"
        d = stats_for_feature.get(key_tuple)
        if d is None:
            d = stats_for_feature.get(str(key_tuple))
        return None if d is None else d.get(field)

    q_low_s  = [ _lookup(k, "q_low")  for k in keys ]
    q_high_s = [ _lookup(k, "q_high") for k in keys ]
    std_s    = [ _lookup(k, "std")    for k in keys ]

    col = feature
    x = df[col].to_numpy(copy=False)

    # fallback quantiles globaux si pas de groupe
    if global_low is None or global_high is None:
        valid = df[col].to_numpy()
        valid = valid[~pd.isna(valid)]
        if valid.size:
            gl = float(np.nanpercentile(valid, WINSOR_Q_LOW*100))
            gh = float(np.nanpercentile(valid, WINSOR_Q_HIGH*100))
        else:
            gl, gh = None, None
    else:
        gl, gh = global_low, global_high

    ql = np.array([gl if (v is None) else v for v in q_low_s], dtype=float)
    qh = np.array([gh if (v is None) else v for v in q_high_s], dtype=float)

    swap = ql > qh
    if swap.any():
        t = ql[swap].copy()
        ql[swap] = qh[swap]
        qh[swap] = t

    df[col] = np.clip(x, ql, qh)

    std_arr = np.array([np.nan if v is None else v for v in std_s], dtype=float)
    df[f"{col}_isconstant"] = ((std_arr < CONST_STD_THRESH) & ~np.isnan(std_arr)).astype("int8")

def _add_wall_flags(df: pd.DataFrame):
    """Ajoute des flags binaires pour wall_opp_share_5/15 selon les seuils."""
    for base_col, thresholds in WALL_THRESHOLDS.items():
        if base_col not in df.columns:
            continue
        for thr in thresholds:
            thr_tag = str(thr).replace(".", "p")
            flag_col = f"{base_col}_gt_{thr_tag}"
            df[flag_col] = (df[base_col] > thr).astype("int8")

def _fix_microprice_bias(df: pd.DataFrame) -> pd.DataFrame:
    col = "microprice_bias"
    if col not in df.columns:
        return df
    x = df[col].astype("float64")
    if MICROPRICE_FIX_MODE == "asinh_then_z":
        x = np.arcsinh(x)
    df[col] = x
    return df

def _prep_heavy_tail(df: pd.DataFrame, cols: set[str]) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            continue
        v = df[c].astype("float64").to_numpy()
        lo, hi = np.nanquantile(v, ROBUST_CLIP_Q[0]), np.nanquantile(v, ROBUST_CLIP_Q[1])
        df[c] = np.arcsinh(np.clip(v, lo, hi))
    return df

def _fs(region: Optional[str]) -> pafs.S3FileSystem:
    return pafs.S3FileSystem(region=region) if region else pafs.S3FileSystem()

def _strip_s3(uri: str) -> str:
    assert uri.startswith("s3://")
    return uri[len("s3://"):]

def _open_s3_output(fs: pafs.S3FileSystem, path_s3: str):
    return fs.open_output_stream(_strip_s3(path_s3))

def _ensure_dir(_fs: pafs.S3FileSystem, _path: str):
    return  # Arrow crée à l'écriture

def _fit_group_guard_stats_train(dataset: ds.Dataset,
                                 filt: Optional[ds.Expression],
                                 sym_id_map: Dict[str, int],
                                 tf_id_map: Dict[str, int],
                                 batch_size: int = 300_000) -> dict:
    """
    Pour chaque feature de WINSOR_FEATURES:
      - q_low (p1), q_high (p99), std
    par groupe TF uniquement. Clés: str(tf_id)
    """
    need_cols = ["tf"] + [c for c in WINSOR_FEATURES if c in dataset.schema.names]
    scanner = dataset.scanner(columns=need_cols, filter=filt, batch_size=batch_size)

    store: Dict[tuple, List[np.ndarray]] = {}  # (tf_id, feature) -> list[np.ndarray]

    for b in scanner.to_batches():
        df = b.to_pandas(types_mapper=pd.ArrowDtype)
        if df.empty:
            continue
        
        _normalize_tf_col_inplace(df)  # ensures '15m' not b'15m' / Arrow scalars
        df["tf_id"] = df["tf"].map(tf_id_map).astype("int32")

        for f in WINSOR_FEATURES:
            if f not in df.columns:
                continue
            g = df[["tf_id", f]].dropna()
            for ti, sub in g.groupby("tf_id", sort=False):
                x = pd.to_numeric(sub[f], errors="coerce").astype("float64").to_numpy()
                x = x[np.isfinite(x)]
                if x.size:
                    store.setdefault((int(ti), f), []).append(x)

    gg_stats: Dict[str, Dict[str, Dict[str, float]]] = {f: {} for f in WINSOR_FEATURES}
    for (ti, f), parts in store.items():
        x = np.concatenate(parts) if parts else np.array([], dtype="float64")
        if x.size == 0:
            continue
        ql = float(np.nanquantile(x, WINSOR_Q_LOW))
        qh = float(np.nanquantile(x, WINSOR_Q_HIGH))
        if ql > qh:
            ql, qh = qh, ql
        std = float(np.nanstd(x))
        gg_stats[f][str(int(ti))] = {"q_low": ql, "q_high": qh, "std": std}
    return gg_stats

# =======================
# Filters
# =======================

def _optional_filter(schema: pa.Schema, symbols: Optional[List[str]], tfs: Optional[List[str]]):
    expr = None
    if symbols and "symbol" in schema.names:
        expr = ds.field("symbol").isin(symbols)
    if tfs and "tf" in schema.names:
        tf_expr = ds.field("tf").isin(tfs)
        expr = tf_expr if expr is None else (expr & tf_expr)
    return expr

def _apply_side_choice(expr: ds.Expression | None, schema: pa.Schema, side_choice: str):
    """
    Retourne une expression de filtre combinant l'existant (expr) et le side:
      - 'shortonly' => side == 'sell'
      - 'longonly'  => side == 'buy'
    Si la colonne 'side' n'existe pas en Stage2, on lève une erreur en mode filtré.
    """
    if side_choice == "both":
        return expr
    if "side" not in schema.names:
        raise ValueError("Colonne 'side' absente en Stage2: impossible d'appliquer --side.")
    want = "sell" if side_choice == "shortonly" else "buy"
    side_expr = (ds.field("side") == want)
    return side_expr if expr is None else (expr & side_expr)

# =======================
# Robust params (TRAIN)
# =======================

def _fit_robust_params_train(dataset: ds.Dataset,
                             filt: Optional[ds.Expression],
                             cols_needed: List[str],
                             batch_size: int) -> pd.DataFrame:
    """
    Calcule p1/p99, médiane, échelle robuste (IQR/1.349) PAR TF (global, pas par symbole).
    Applique log1p/ASINH/clip lourd avant stats.
    """
    avail = set(dataset.schema.names)
    proj = [c for c in (["tf"] + [c for c in cols_needed if c != "tf"]) if c in avail]
    scanner = dataset.scanner(columns=proj, filter=filt, batch_size=batch_size)

    feat_for_stats = sorted((set(ZSCORE_FEATURES) | set(LOG1P_FEATURES) | set(LATENCY_FEATURES)) & avail)
    q_lo, q_hi = ROBUST_CLIP_Q
    store: Dict[tuple, List[np.ndarray]] = {}  # (tf, feature) -> arrays

    for batch in scanner.to_batches():
        df = batch.to_pandas(types_mapper=pd.ArrowDtype)
        df["tf"] = df["tf"].map(_tf_key)
        _normalize_tf_col_inplace(df) 
        if "tf" not in df.columns:
            raise ValueError("Column 'tf' must exist in stage2 files.")
        # pré-transfos
        for f in LOG1P_FEATURES:
            if f in df.columns:
                v = pd.to_numeric(df[f], errors="coerce").astype("float64")
                df[f] = np.log1p(v.clip(lower=0))
        for f in LATENCY_FEATURES:
            if f in df.columns:
                v = pd.to_numeric(df[f], errors="coerce").astype("float64")
                df[f] = np.log1p(v.clip(lower=0))
        _apply_asinh_inplace(df)
        df = _fix_microprice_bias(df)
        df = _prep_heavy_tail(df, HEAVY_TAIL_COLS)

        for tf_key, g in df.groupby("tf", sort=False):  # scalar key
            for f in feat_for_stats:
                if f not in g.columns:
                    continue
                x = pd.to_numeric(g[f], errors="coerce").astype("float64").to_numpy()
                x = x[np.isfinite(x)]
                if x.size:
                    store.setdefault((_tf_key(tf_key), f), []).append(x)

    rows = []
    for (tf, f), parts in store.items():
        x = np.concatenate(parts) if parts else np.array([], dtype="float64")
        if x.size == 0:
            rows.append((tf, f, np.nan, np.nan, 0.0, 1.0)); continue
        q1 = float(np.nanquantile(x, q_lo))
        q99 = float(np.nanquantile(x, q_hi))
        xc = np.clip(x, q1, q99)
        med = float(np.nanmedian(xc))
        q25 = float(np.nanquantile(xc, 0.25))
        q75 = float(np.nanquantile(xc, 0.75))
        iqr = max(q75 - q25, 0.0)
        scale = (iqr / 1.349) if iqr > 0 else 0.0
        if not np.isfinite(scale) or scale < ROBUST_EPS:
            scale = 1.0
        scale = max(scale, ROBUST_SCALE_FLOOR)
        rows.append(( _norm_str(tf), f, q1, q99, med, scale))
    
    if not store:
        # Helpful debug: what did the dataset actually expose?
        avail = set(dataset.schema.names)
        raise RuntimeError(
            "Robust-stats scan found no feature data. "
            f"Schema had {len(avail)} cols, example: {sorted(list(avail))[:20]}. "
            "Check that Stage2 has the expected feature columns "
            "(e.g., 'spread_bps_entry','obi_5','microprice_bias', etc.)."
        )

    return pd.DataFrame(rows, columns=["tf","feature","q1","q99","median","scale"])

# =======================
# Normalization + encoding
# =======================

def _apply_norm(df: pd.DataFrame, 
                params_map: Dict[str, Dict[str, Tuple[float, float, float, float]]]) -> pd.DataFrame:
    df = df.copy()

    # pré-transfos log1p
    for f in LOG1P_FEATURES:
        if f in df.columns:
            v = pd.to_numeric(df[f], errors="coerce").astype("float64")
            df[f] = np.log1p(v.clip(lower=0))
    for f in LATENCY_FEATURES:
        if f in df.columns:
            v = pd.to_numeric(df[f], errors="coerce").astype("float64")
            df[f] = np.log1p(v.clip(lower=0))

    _apply_asinh_inplace(df)
    df = _fix_microprice_bias(df)
    df = _prep_heavy_tail(df, HEAVY_TAIL_COLS)

    if "tf" not in df.columns:
        raise ValueError("Column 'tf' must exist")
    
    df["tf"] = df["tf"].map(_tf_key)

    # Masques + imputation neutre
    for f, neutral in IMPUTE_NEUTRAL.items():
        if f in df.columns:
            s = pd.to_numeric(df[f], errors="coerce").astype("float64")
            mask = ~np.isfinite(s)
            df[f"{f}_isnan"] = mask.astype("int8")
            s[mask] = neutral
            df[f] = s

    # Robust z-score par (symbol, TF)
    groups = df.groupby("tf", sort=False).groups
    for tf_key, idx in groups.items():
        pm = params_map.get(_tf_key(tf_key))
        if not pm:
            continue
        for f in set(ZSCORE_FEATURES) | set(LOG1P_FEATURES) | set(LATENCY_FEATURES):
            if f in df.columns and f in pm:
                q1, q99, med, scale = pm[f]
                x = pd.to_numeric(df.loc[idx, f], errors="coerce").astype("float64")
                x = np.clip(x, q1, q99)
                denom = scale if (np.isfinite(scale) and scale >= ROBUST_EPS) else 1.0
                df.loc[idx, f] = (x - med) / denom

    # side_num
    if "side" in df.columns:
        df["side_num"] = df["side"].map({"buy": 1, "sell": -1}).fillna(0).astype("int8")
    else:
        df["side_num"] = 0

    # label en int
    df["Y"] = pd.to_numeric(df.get("Y", 0), errors="coerce").fillna(0).astype("int64")

    # garde-fous (NaN → 0 après normalisation)
    for c in set(LOG1P_FEATURES) | set(ZSCORE_FEATURES):
        if c in df.columns:
            col = pd.to_numeric(df[c], errors="coerce").astype("float64")
            col[~np.isfinite(col)] = 0.0
            df[c] = col

    # Pare-chocs post-zscore
    POST_Z_CLIP = {
        "spread_bps_entry": 4.0,
        "quote_churn_10s":  6.0,
        "executed_vs_added_ratio": 6.0,
    }
    for c, lim in POST_Z_CLIP.items():
        if c in df.columns:
            x = pd.to_numeric(df[c], errors="coerce").astype("float64")
            df[c] = np.clip(x, -lim, +lim)

    return df

# =======================
# CSV writer helpers
# =======================

def _build_id_maps(dataset: ds.Dataset, filt: Optional[ds.Expression], batch_size: int = 250_000) -> Tuple[Dict[str, int], Dict[str, int]]:
    cols = [c for c in ("symbol", "tf") if c in dataset.schema.names]
    scanner = dataset.scanner(columns=cols, filter=filt, batch_size=batch_size)
    sym, tf = set(), set()
    for b in scanner.to_batches():
        df = b.to_pandas()
        if "symbol" in df: sym.update(df["symbol"].dropna().map(_norm_str).unique().tolist())
        if "tf" in df:
            tf.update(df["tf"].dropna().map(_tf_key).unique().tolist())
    return ({s: i for i, s in enumerate(sorted(sym))},
            {t: i for i, t in enumerate(sorted(tf))})

def _rows_to_csv_bytes(arr2d: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.savetxt(buf, arr2d, delimiter=",", fmt="%.6g")
    return buf.getvalue()

def _write_csv_shards(fs: pafs.S3FileSystem, base_out: str, iterator, rows_per_shard: int) -> int:
    part = 0
    written = 0
    acc: List[np.ndarray] = []
    cur = 0

    def _dump(path, arr2d):
        with _open_s3_output(fs, path) as out:
            out.write(_rows_to_csv_bytes(arr2d))

    for X in iterator:
        if X.size == 0:
            continue
        acc.append(X); cur += X.shape[0]

        while cur >= rows_per_shard:
            big = np.vstack(acc)
            head, tail = big[:rows_per_shard], big[rows_per_shard:]
            path = f"{base_out.rstrip('/')}/part-{part:05d}.csv"
            _dump(path, head)
            written += head.shape[0]; part += 1
            acc = [tail] if tail.size else []; cur = tail.shape[0] if tail.size else 0

    if acc:
        big = np.vstack(acc)
        path = f"{base_out.rstrip('/')}/part-{part:05d}.csv"
        _dump(path, big)
        written += big.shape[0]
    return written

def _write_weights_shards(fs: pafs.S3FileSystem, base_out: str, it_pairs, rows_per_shard: int) -> int:
    part = 0
    written = 0
    acc: List[np.ndarray] = []
    cur = 0

    def _dump(path, arr1d):
        v = arr1d.reshape(-1, 1)
        with _open_s3_output(fs, path) as out:
            out.write(_rows_to_csv_bytes(v))

    for _, w in it_pairs:
        if w.size == 0:
            continue
        acc.append(w.astype("float32")); cur += w.size

        while cur >= rows_per_shard:
            big = np.concatenate(acc, axis=0)
            head, tail = big[:rows_per_shard], big[rows_per_shard:]
            path = f"{base_out.rstrip('/')}/part-{part:05d}.csv"
            _dump(path, head)
            written += head.shape[0]; part += 1
            acc = [tail] if tail.size else []; cur = tail.shape[0] if tail.size else 0

    if acc:
        big = np.concatenate(acc, axis=0)
        path = f"{base_out.rstrip('/')}/part-{part:05d}.csv"
        _dump(path, big)
        written += big.shape[0]
    return written

# =======================
# Const-feature detector (TRAIN)
# =======================

def _constant_features_on_train_tf(dataset: ds.Dataset,
                                   filt: Optional[ds.Expression],
                                   cols: List[str],
                                   params_map: Dict[str, Dict[str, Tuple[float, float, float, float]]],
                                   batch_size: int = 200_000,
                                   eps: float = 1e-6) -> List[str]:
    if not cols:
        return []
    proj = ["tf"] + cols
    scanner = dataset.scanner(columns=proj, filter=filt, batch_size=batch_size)
    stats = {c: [] for c in cols}

    for b in scanner.to_batches():
        df = b.to_pandas(types_mapper=pd.ArrowDtype)
        df["tf"] = df["tf"].map(_tf_key)        
        if "tf" not in df.columns:
            continue
        # pré-transfos...
        for f in LOG1P_FEATURES:
            if f in df.columns:
                v = pd.to_numeric(df[f], errors="coerce").astype("float64")
                df[f] = np.log1p(v.clip(lower=0))
        for f in LATENCY_FEATURES:
            if f in df.columns:
                v = pd.to_numeric(df[f], errors="coerce").astype("float64")
                df[f] = np.log1p(v.clip(lower=0))

        for tf_key, g in df.groupby("tf", sort=False):  # scalar key    
            pm = params_map.get(_tf_key(tf_key), {})
            for c in cols:
                if c not in g.columns:
                    continue
                x = pd.to_numeric(g[c], errors="coerce").astype("float64").to_numpy()
                if c in pm:
                    q1, q99, _, _ = pm[c]
                    x = np.clip(x, q1, q99)
                x = x[np.isfinite(x)]
                v = float(x.std(ddof=0)) if x.size else 0.0
                if np.isfinite(v):
                    stats[c].append(v)

    consts = []
    for c, arr in stats.items():
        if not arr:
            consts.append(c)
        else:
            med = float(np.median(arr))
            if not np.isfinite(med) or med < eps:
                consts.append(c)
    return consts

# =======================
# CLI & pipeline
# =======================

def _dataset_from_split(fs: pafs.S3FileSystem, src_root: str, split: str) -> ds.Dataset:
    base = _strip_s3(f"{src_root.rstrip('/')}/{split}")
    return ds.dataset(base, filesystem=fs, format="parquet", partitioning="hive")

def parse_args():
    ap = argparse.ArgumentParser(
        description="Stage2(train/val/test) → CSV normalisés XGBoost (scalers & guards au niveau TF global)."
    )
    ap.add_argument("--src-root", default="s3://tradebot-config-tokyo/data/stage2/v2")
    ap.add_argument("--dst-root", required=True, help="s3://…/data/stage3-xgb/v1")
    ap.add_argument("--aws-region", default="ap-northeast-1")
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--tfs", nargs="*", default=None)
    ap.add_argument("--batch-size", type=int, default=200_000)
    ap.add_argument("--rows-per-shard", type=int, default=1_000_000)
    ap.add_argument("--save-mappings", default=True,
                    help="Écrit _meta/mappings.json & _meta/columns.json basés sur TRAIN")
    ap.add_argument("--side", choices=["both","shortonly","longonly"], default="both",
                    help="Filtre du flux Stage2 par side (sell→shortonly, buy→longonly). "
                         "Le split est appliqué avant tout fit/normalisation.")
    ap.add_argument("--auto_branch_suffix", action="store_true",
                    help="Si activé et --side≠both, ajoute automatiquement -shortonly / -longonly à --dst-root.")
    return ap.parse_args()

def _make_wall_flag_names() -> List[str]:
    names = []
    for base_col, thresholds in WALL_THRESHOLDS.items():
        for thr in thresholds:
            names.append(f"{base_col}_gt_{str(thr).replace('.','p')}")
    return names

def _pre_normalize_feature_engineering(df_chunk: pd.DataFrame, gg_stats: dict) -> pd.DataFrame:
    # winsor + _isconstant par groupe
    for feat in WINSOR_FEATURES:
        if feat in df_chunk.columns:
            _apply_winsor_and_constant_mask(
                df_chunk,
                feature=feat,
                stats_for_feature=gg_stats.get(feat, {}),
                global_low=None, global_high=None
            )
    # flags mur
    _add_wall_flags(df_chunk)
    return df_chunk

def _detect_constant_flags_train(dataset: ds.Dataset,
                                 filt: Optional[ds.Expression],
                                 flag_cols: List[str],
                                 gg_stats: dict,
                                 tf_id_map: Dict[str, int],
                                 batch_size: int = 200_000) -> List[str]:
    """Scanne TRAIN pour repérer les FLAG_COLS constants (0% ou 100% de 1).
       On recalcule les flags via _pre_normalize_feature_engineering (winsor/_isconstant + walls)."""
    if not flag_cols:
        return []
    need = {"tf"} | set(WINSOR_FEATURES) | set(WALL_THRESHOLDS.keys())
    need = [c for c in need if c in dataset.schema.names] + ["tf"]  # tf déjà inclus

    scanner = dataset.scanner(columns=sorted(set(need)), filter=filt, batch_size=batch_size)
    totals = {c: 0 for c in flag_cols}
    ones   = {c: 0 for c in flag_cols}

    for b in scanner.to_batches():
        df = b.to_pandas(types_mapper=pd.ArrowDtype)
        if df.empty: 
            continue
        # tf_id pour guards
        df["tf"] = df["tf"].map(_tf_key)
        df["tf_id"] = df["tf"].map(tf_id_map).fillna(-1).astype("int32")
        # recalcul des flags
        df = _pre_normalize_feature_engineering(df, gg_stats)

        # accumulateurs
        for c in flag_cols:
            if c in df.columns:
                s = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
                totals[c] += int(s.shape[0])
                ones[c]   += int((s > 0).sum())

    const = []
    for c in flag_cols:
        n = totals.get(c, 0)
        k = ones.get(c, 0)
        if n == 0:
            continue
        if k == 0 or k == n:
            const.append(c)
    return const

def main():
    args = parse_args()
    fs = _fs(args.aws_region)

    # Datasets
    ds_train = _dataset_from_split(fs, args.src_root, "train")
    schema = ds_train.schema
    try:
        ds_val = _dataset_from_split(fs, args.src_root, "val")
    except Exception:
        ds_val = None
    try:
        ds_test = _dataset_from_split(fs, args.src_root, "test")
    except Exception:
        ds_test = None

    # Filtres (symbols/tfs) puis split par side AVANT toute stat/normalisation
    f_train = _optional_filter(schema, args.symbols, args.tfs)
    f_train = _apply_side_choice(f_train, schema, args.side)
    if ds_val:
        f_val = _optional_filter(ds_val.schema, args.symbols, args.tfs)
        f_val = _apply_side_choice(f_val, ds_val.schema, args.side)
    else:
        f_val = None
    if ds_test:
        f_test = _optional_filter(ds_test.schema, args.symbols, args.tfs)
        f_test = _apply_side_choice(f_test, ds_test.schema, args.side)
    else:
        f_test = None

    # Comptes
    n_tr = ds_train.count_rows(filter=f_train)
    n_va = ds_val.count_rows(filter=f_val) if ds_val else 0
    n_te = ds_test.count_rows(filter=f_test) if ds_test else 0
    print(f"[stage2] rows → TRAIN:{n_tr} | VAL:{n_va} | TEST:{n_te}")
    if n_tr == 0:
        print("⛔ TRAIN vide — rien à fitter."); sys.exit(2)

    # Colonnes à lire
    schema_names = set(ds_train.schema.names)
    if ds_val:  schema_names |= set(ds_val.schema.names)
    if ds_test: schema_names |= set(ds_test.schema.names)

    wanted = sorted(
        (set(ALWAYS_KEEP)
         | set(LOG1P_FEATURES)
         | set(ZSCORE_FEATURES)
         | set(LATENCY_FEATURES)
         | set(WEIGHT_ONLY_COLS))   # ← lire pour poids, pas pour features
        & schema_names
    )

    # Suffixe automatique de la destination (pratique pour éviter les collisions)
    dst_root = args.dst_root
    if args.auto_branch_suffix and args.side in ("shortonly","longonly"):
        suf = "-shortonly" if args.side == "shortonly" else "-longonly"
        if not dst_root.rstrip("/").endswith(suf):
            dst_root = dst_root.rstrip("/") + suf
    print(f"[dst] output root: {dst_root}")

    # 1) Fit robust params sur TRAIN (par TF)
    print("Fitting robust TF-level params on TRAIN…")
    _feat_cols_for_stats = list(
        (set(ZSCORE_FEATURES) | set(LOG1P_FEATURES) | set(LATENCY_FEATURES))
    )
    
    rob = _fit_robust_params_train(ds_train, f_train, wanted, batch_size=args.batch_size)

    # normalize tf column to plain str
    rob["tf"] = rob["tf"].map(_tf_key)

    params_map: Dict[str, Dict[str, Tuple[float, float, float, float]]] = {}
    for tf_key, grp in rob.groupby("tf"):  # << scalar key, not ["tf"]
        # belt & suspenders if some pandas version still gives a tuple
        if isinstance(tf_key, (tuple, list)) and len(tf_key) == 1:
            tf_key = tf_key[0]
        tf_str = _tf_key(tf_key)           # ensures '15m', '1h', ...
        params_map[tf_str] = {
            feat: (float(q1), float(q99), float(med), float(scale))
            for feat, q1, q99, med, scale in grp[["feature","q1","q99","median","scale"]].itertuples(index=False, name=None)
        }

    # --- Fallback de scaler pour features indispensables manquantes ---
    NEED_ALWAYS = {"microprice_bias"}  # ajoute-en d’autres si besoin
    seen_feats = set(rob["feature"].astype(str).tolist())
    DEFAULT_SCALER = (0.0, 0.0, 0.0, 1.0)  # (q1, q99, median, scale)

    for tf_key, pm in params_map.items():
        for f in NEED_ALWAYS:
            # on n'injecte que si la feature existe globalement (vue pendant le fit)
            if (f not in pm) and (f in seen_feats):
                pm[f] = DEFAULT_SCALER

    missing_by_tf = {
        tf: sorted(list(NEED_ALWAYS - set(pm.keys())))
        for tf, pm in params_map.items()
        if not NEED_ALWAYS.issubset(pm.keys())
    }
    if any(missing_by_tf.values()):
        print(f"[scaler-fallback] Injected defaults for NEED_ALWAYS per TF (0/0/0/1). "
            f"Missing after inject (should be empty): {missing_by_tf}")
        
    _has_mp = any(("microprice_bias" in pm) for pm in params_map.values())
    if not _has_mp:
        print("⚠️ WARN: aucun param trouvé pour 'microprice_bias' dans params_map "
              "(il ne sera pas z-scoré). Vérifie _fit_robust_params_train → feat_for_stats / colonnes dispo.")

    # 2) ID mappings (TRAIN)
    sym_id, tf_id = _build_id_maps(ds_train, f_train)
    all_tfs_seen = set(tf_id.keys())  # canonical from Stage2
    feat_for_stats = sorted(set(ZSCORE_FEATURES) | set(LOG1P_FEATURES) | set(LATENCY_FEATURES))

    missing = all_tfs_seen - set(params_map.keys())
    if missing:
        print("[debug] tfs in params_map:", sorted(params_map.keys()))
        print("[debug] tfs in tf_id:", sorted(tf_id.keys()))
        raise RuntimeError(f"Missing scaler_stats for TFs: {sorted(missing)} "
                       f"(clé TF non normalisée ?)")

    # 3) Fit & save group-guard stats
    print("Fitting group-guard stats (winsor + isconstant) on TRAIN…")
    gg_stats = _fit_group_guard_stats_train(ds_train, f_train, sym_id, tf_id, batch_size=args.batch_size)
    _save_group_stats(gg_stats, fs, dst_root)
    print(f"→ saved group-guard stats to {dst_root.rstrip('/')}/{META_GG_STATS}")
    gg_stats_loaded = gg_stats  # on a déjà l'objet en RAM

    # 4) Choix des colonnes / drop constantes
    _all_feats = list(dict.fromkeys(LOG1P_FEATURES + ZSCORE_FEATURES))
    mask_cols = [f"{f}_isnan" for f in IMPUTE_NEUTRAL.keys()]
    _all_feats_with_masks = list(dict.fromkeys(_all_feats + mask_cols))

    candidates = [c for c in _all_feats if c in wanted]
    const_on_train = set(_constant_features_on_train_tf(
        ds_train, f_train, candidates, params_map, batch_size=args.batch_size, eps=1e-6
    ))

    FORCE_KEEP = {"slope_bid_15"}
    DROP_FEATURES = set(const_on_train) - FORCE_KEEP
    DROP_SUFFIXES = ("_missing",)

    feature_cols = [
        c for c in _all_feats_with_masks
        if (c in (set(wanted) | set(mask_cols)))
        and (c not in DROP_FEATURES)
        and not any(c.endswith(suf) for suf in DROP_SUFFIXES)
    ]

    if args.side in ("shortonly","longonly"):
        BASES_TO_DROP = {
            "obi_5","obi_15","microprice_bias",
            "slope_bid_5","slope_ask_5","slope_bid_15","slope_ask_15",
            "aggr_ratio_10s","net_delta_15s","mid_jump_bps_3s",
        }
        feature_cols = [c for c in feature_cols if c not in BASES_TO_DROP]

    # étendre la grille avec _isconstant + flags murs
    CONST_MASK_COLS = [f"{f}_isconstant" for f in WINSOR_FEATURES if f in wanted]
    CONST_MASK_COLS = [c for c in CONST_MASK_COLS if (c.replace("_isconstant","") in wanted)]
    WALL_FLAG_COLS  = list(_make_wall_flag_names())
    feature_cols = list(dict.fromkeys(feature_cols + CONST_MASK_COLS + WALL_FLAG_COLS))


    # --- FLAGS BINAIRES qui auront besoin d'un scaler "binaire" (0/1) ---
    FLAG_COLS = []
    FLAG_COLS += [f"{f}_isnan" for f in IMPUTE_NEUTRAL.keys() if f"{f}_isnan" in feature_cols]
    FLAG_COLS += [f"{f}_isconstant" for f in WINSOR_FEATURES if f"{f}_isconstant" in feature_cols]
    FLAG_COLS += [c for c in feature_cols if "_gt_" in c]
    FLAG_COLS = sorted(dict.fromkeys(FLAG_COLS))  # unique + stable order

    # --- Drop des flags constants sur TRAIN (pré-pass dédié) ---
    if FLAG_COLS:
        const_flags = _detect_constant_flags_train(
            ds_train, f_train, FLAG_COLS, gg_stats_loaded, tf_id, batch_size=args.batch_size
        )
        if const_flags:
            print(f"[flags] Dropping constant flags on TRAIN: {sorted(const_flags)}")
            feature_cols = [c for c in feature_cols if c not in const_flags]
            FLAG_COLS    = [c for c in FLAG_COLS    if c not in const_flags]

    out_order = ["Y", "side_num"] + feature_cols
    print(f"[drop] constant features on TRAIN (tf-level): {sorted(const_on_train)}")
    print(f"[cols] CSV order: {len(out_order)} columns")

    # --- FLAGS BINAIRES qui auront besoin d'un scaler "binaire" (0/1) ---
    FLAG_COLS = []
    # masques de NaN issus d'IMPUTE_NEUTRAL
    FLAG_COLS += [f"{f}_isnan" for f in IMPUTE_NEUTRAL.keys() if f"{f}_isnan" in feature_cols]
    # masques _isconstant issus des guards WINSOR_FEATURES
    FLAG_COLS += [f"{f}_isconstant" for f in WINSOR_FEATURES if f"{f}_isconstant" in feature_cols]
    # flags mur *_gt_*
    FLAG_COLS += [c for c in feature_cols if "_gt_" in c]

    FLAG_COLS = sorted(dict.fromkeys(FLAG_COLS))  # unique + stable order

    # Accumulateurs pour les médianes par tf (sur TRAIN uniquement)
    _tf_flag_sum: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))  # tf -> flag -> somme de 1
    _tf_count: dict[str, int] = defaultdict(int)                                     # tf -> nb de lignes

    # --- QA per-TF accumulateur (mean/std) sur un sous-ensemble clé ---
    QA_KEYS = ["entry","spread_bps_entry","obi_5","obi_15","mid_jump_bps_3s","atr_bps","executed_vs_added_ratio"]
    _per_tf_sum   = {}
    _per_tf_sumsq = {}
    _per_tf_cnt   = {}

    def _qa_accumulate_per_tf(df_norm: pd.DataFrame):
        if "tf" not in df_norm.columns:
            return
        keys_avail = [k for k in QA_KEYS if k in df_norm.columns]
        if not keys_avail:
            return
        for tf_val, g in df_norm.groupby("tf", sort=False):
            tf = str(tf_val)
            _per_tf_cnt[tf] = _per_tf_cnt.get(tf, 0) + int(g.shape[0])
            for k in keys_avail:
                v = pd.to_numeric(g[k], errors="coerce").astype("float64").to_numpy()
                v = v[np.isfinite(v)]
                if v.size == 0:
                    continue
                _per_tf_sum.setdefault(tf, {}).setdefault(k, 0.0)
                _per_tf_sumsq.setdefault(tf, {}).setdefault(k, 0.0)
                _per_tf_sum[tf][k]   += float(v.sum())
                _per_tf_sumsq[tf][k] += float((v * v).sum())

    def _qa_finalize_and_print(dst_root: str, split_tag: str):
        import math, json
        rows = []
        for tf, n in sorted(_per_tf_cnt.items()):
            for k in QA_KEYS:
                s  = _per_tf_sum.get(tf, {}).get(k, 0.0)
                s2 = _per_tf_sumsq.get(tf, {}).get(k, 0.0)
                if n > 0:
                    mean = s / n
                    var  = max((s2 / n) - (mean * mean), 0.0)
                    std  = math.sqrt(var)
                else:
                    mean, std = float("nan"), float("nan")
                rows.append((tf, k, n, mean, std))
        if rows:
            df = pd.DataFrame(rows, columns=["tf","feature","n","mean","std"])
            print("\n[qa] per-TF mean/std (extraits):")
            with pd.option_context("display.max_columns", None, "display.width", 160):
                # on limite l’affichage pour rester lisible
                print(df.head(30).to_string(index=False))
            # on sauve une version complète pour audit
            path = f"{dst_root.rstrip('/')}/_meta/qa_per_tf_{split_tag}.json"
            with _open_s3_output(_fs(None) if isinstance(dst_root, str) else fs, path) as out:
                out.write(df.to_json(orient="records").encode("utf-8"))

    # 5) Scanner/écrire
    def _iter_batches(dataset: ds.Dataset, split: str, filt: Optional[ds.Expression]):
        scanner = dataset.scanner(columns=wanted, filter=filt, batch_size=args.batch_size)
        total = 0
        for b in scanner.to_batches():
            df = b.to_pandas(types_mapper=pd.ArrowDtype)
            _normalize_tf_col_inplace(df)

            # Drops défensifs (anti-fuite) AVANT normalisation/features
            miss = [c for c in df.columns if c.endswith("_missing")]
            if miss:
                df = df.drop(columns=miss, errors="ignore")

            # IDs d’abord (nécessaires aux guards)
            df["tf_id"] = df["tf"].map(tf_id).fillna(-1).astype("int32")

            # guards/flags AVANT normalisation
            df = _pre_normalize_feature_engineering(df, gg_stats_loaded)

            # Normalisation features (log1p/asinh/robust z, masks, etc.)
            df = _apply_norm(df, params_map)

            # QA: accumule per-TF sur les features clés
            _qa_accumulate_per_tf(df)

            # side_num s'il n'existe pas (sécurité)
            if "side_num" not in df.columns:
                df["side_num"] = df["side"].map({"buy": 1, "sell": -1}).fillna(0).astype("int8")

            # === LABEL (GO/NO-GO) ===
            if "side_num" not in df.columns:
                raise ValueError("side_num manquant (buy→+1, sell→-1).")

            y_tri = pd.to_numeric(df["Y"], errors="coerce").fillna(0).astype("int8")
            sgn   = df["side_num"].astype("int8")

            # GO = (direction * side == +1), i.e. "dans le sens du trade"
            y_dir  = (y_tri * sgn).astype("int8")
            df["Y"] = (y_dir == 1).astype("int8")

            # === Pondérations par-ligne ===
            if split == "train":

                w_tf = 1.0 / df.groupby(["tf"])["Y"].transform("count").clip(lower=1)
                w_tf /= w_tf.mean() if w_tf.mean()>0 else 1.0
                w_tf = w_tf.clip(0.5, 3.0)
                
                # (B) incertitude via spread
                if "spread_bps_entry" in df.columns and df["spread_bps_entry"].notna().any():
                    spread_q90 = df["spread_bps_entry"].quantile(0.90)
                    if pd.isna(spread_q90) or spread_q90 <= 0:
                        w_uncert = pd.Series(1.0, index=df.index)
                    else:
                        w_uncert = 1.0 / (1.0 + (df["spread_bps_entry"].clip(lower=0) / spread_q90))
                        w_uncert = w_uncert.clip(0.5, 2.0)
                else:
                    w_uncert = pd.Series(1.0, index=df.index)

                w = 0.7 * w_tf + 0.3 * w_uncert

                # 🟩 BOOST LONGS POSITIFS (au lieu d’oversampling physique)
                if args.side == "longonly":
                    BOOST = 1.5         # valeur initiale, on pourra tuner
                    ybin = df["Y"].to_numpy()
                    w = np.where(ybin == 1, w * BOOST, w)

                m = float(w.mean()) if np.isfinite(w.mean()) and w.mean() > 0 else 1.0
                df["weight"] = (w / m).astype("float32")
            else:
                df["weight"] = 1.0
            
            # --- Agrégation des flags binaires pour scaler binaire (TRAIN only) ---
            if split == "train" and FLAG_COLS:
                # On agrège par tf dans CE lot, puis on cumule
                for tf_val, sub in df.groupby("tf", sort=False):
                    tf_str = _norm_str(tf_val) 
                    _tf_count[tf_str] += int(sub.shape[0])
                    for c in FLAG_COLS:
                        if c in sub.columns:
                            # On force binaire: toute valeur non nulle compte comme 1
                            s = pd.to_numeric(sub[c], errors="coerce").fillna(0.0)
                            _tf_flag_sum[tf_str][c] += int((s > 0).sum())

            # --- META ROWS (avant drop des colonnes)
            # row_id stable: symbol|tf|t
            df["row_id"] = df[["symbol","tf","t"]].astype(str).agg("|".join, axis=1)
            meta_keep = df[["row_id","symbol","tf","t","Y","side_num","weight"]].copy()
            _append_meta_parquet(fs, f"{dst_root.rstrip('/')}/_meta/splits_parquet", split, meta_keep)

            # Nettoyage anti-fuite & RAM: on peut dropper les colonnes poids-only
            df.drop(columns=list(WEIGHT_ONLY_COLS & set(df.columns)), inplace=True, errors="ignore")
            # plus EXCLUDE_COLS (ex: THRESH_BPS)
            df.drop(columns=list(EXCLUDE_COLS & set(df.columns)), inplace=True, errors="ignore")

            # ensure columns de la grille de sortie (sans weight)
            for c in out_order:
                if c not in df.columns:
                    df[c] = 0 if c in ("side_num","Y") else 0.0

            # to arrays
            y   = pd.to_numeric(df["Y"], errors="coerce").fillna(0).astype("int64").to_numpy()
            sgn = df["side_num"].astype("int32").to_numpy()
            Xf  = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype="float32")
            wgt = pd.to_numeric(df["weight"], errors="coerce").fillna(1.0).astype("float32").to_numpy()

            # Schéma CSV principal: [Y, side_num, features…] (PAS de weight inline)
            X = np.column_stack([y, sgn, Xf]).astype("float32", copy=False)
            X = np.nan_to_num(X, copy=False, posinf=0.0, neginf=0.0)
            total += X.shape[0]
            yield (X, wgt)   # 👈 on renvoie un PAIR (X, wgt)

        print(f"  → {split}: produced rows≈{total}")

    # 6) Write TRAIN
    train_out   = f"{dst_root.rstrip('/')}/train"
    train_w_out = f"{dst_root.rstrip('/')}/train_weight"
    print("Writing TRAIN CSV shards…")
    it_train = list(_iter_batches(ds_train, "train", f_train))
    ntrain   = _write_csv_shards(fs, train_out, (X for X, _ in it_train), args.rows_per_shard)
    ntrain_w = _write_weights_shards(fs, train_w_out, it_train, args.rows_per_shard)
    print(f"✅ TRAIN written: {ntrain} rows  | weights: {ntrain_w}")
    _qa_finalize_and_print(dst_root, "train")

    # === After TRAIN write, compute and save scale_pos_weight ===
    pos_count = 0
    neg_count = 0
    for X, _ in it_train:
        y = X[:, 0]  # col 0 = Y
        pos_count += int((y == 1).sum())
        neg_count += int((y == 0).sum())

    posw = float(neg_count) / float(pos_count) if pos_count > 0 else 1.0
    print(f"[meta] scale_pos_weight (train) = {posw:.4f}")

    meta_dir = f"{dst_root.rstrip('/')}/_meta"
    with _open_s3_output(fs, f"{meta_dir}/train_pos_weight.json") as f:
        f.write(json.dumps({
            "pos_weight": posw,
            "pos_count": pos_count,
            "neg_count": neg_count
        }, indent=2).encode("utf-8"))

    # 7) VALIDATION
    if ds_val and n_va > 0:
        val_out   = f"{dst_root.rstrip('/')}/validation"
        val_w_out = f"{dst_root.rstrip('/')}/validation_weight"
        print("Writing VALIDATION CSV shards…")
        it_val = list(_iter_batches(ds_val, "validation", f_val))
        nval   = _write_csv_shards(fs, val_out, (X for X, _ in it_val), args.rows_per_shard)
        nval_w = _write_weights_shards(fs, val_w_out, it_val, args.rows_per_shard)
        print(f"✅ VALIDATION written: {nval} rows  | weights: {nval_w}")
        _qa_finalize_and_print(dst_root, "validation")
    else:
        print("ℹ️ VALIDATION empty → skipped")

    # 8) TEST
    if ds_test and n_te > 0:
        test_out   = f"{dst_root.rstrip('/')}/test"
        test_w_out = f"{dst_root.rstrip('/')}/test_weight"
        print("Writing TEST CSV shards…")
        it_test = list(_iter_batches(ds_test, "test", f_test))
        ntest   = _write_csv_shards(fs, test_out, (X for X, _ in it_test), args.rows_per_shard)
        ntest_w = _write_weights_shards(fs, test_w_out, it_test, args.rows_per_shard)
        print(f"✅ TEST written: {ntest} rows  | weights: {ntest_w}")
        _qa_finalize_and_print(dst_root, "test")
    else:
        print("ℹ️ TEST empty → skipped")

    # 9) mappings + columns.json
    if args.save_mappings:
        meta_dir = f"{dst_root.rstrip('/')}/_meta"
        _ensure_dir(fs, meta_dir)

        # Étendre rob avec des lignes pour chaque FLAG_COLS manquant
        rob_ext = rob.copy()

        # Index existants (tf, feature) pour savoir ce qui manque
        existing = set((str(r.tf), r.feature) for r in rob_ext.itertuples(index=False))

        extra_rows = []
        if FLAG_COLS:
            # Liste de tous les tf vus dans params_map (donc vus pendant le fit)
            all_tf_vals = sorted(params_map.keys())
            for tf_str in all_tf_vals:
                n_tf = _tf_count.get(tf_str, 0)
                for flag in FLAG_COLS:
                    if (tf_str, flag) in existing:
                        continue
                    # médiane binaire ≈ 1 si fréquence >= 0.5
                    ones = _tf_flag_sum[tf_str].get(flag, 0)
                    med = 1.0 if (n_tf > 0 and (ones * 2 >= n_tf)) else 0.0
                    extra_rows.append({
                        "tf": tf_str,
                        "feature": flag,
                        "q1": 0.0,
                        "q99": 1.0,
                        "median": float(med),
                        "scale": 0.5,
                    })

        if extra_rows:
            rob_ext = pd.concat([rob_ext, pd.DataFrame(extra_rows)], axis=0, ignore_index=True)

        rob_ext["tf"] = rob_ext["tf"].map(_norm_str)

        with _open_s3_output(fs, f"{meta_dir}/scaler_stats.json") as f:
            f.write(rob_ext.to_json(orient="records").encode("utf-8"))

        with _open_s3_output(fs, f"{meta_dir}/mappings.json") as f:
            f.write(json.dumps({"tf": tf_id, "symbol": sym_id}, indent=2).encode("utf-8"))

        with _open_s3_output(fs, f"{meta_dir}/columns.json") as f:
            f.write(json.dumps({
                "features": feature_cols,
                "label": "Y",
                "side": "side_num"
                }, indent=2).encode("utf-8"))

        print("📝 saved _meta/scaler_stats.json & _meta/columns.json")

    print("Done.")

if __name__ == "__main__":
    main()