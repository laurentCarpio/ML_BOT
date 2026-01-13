# -*- coding: utf-8 -*-
"""
Inference validator for a frozen XGBoost model release — Stage3_GO only.

Stage3_GO contract:
  - batch CSV shards (no header): [label] + features
  - no side_num, no tf, no meta parquet
  - per-pocket thresholds and EV policy are NOT supported here.
"""

import os
import io
import json
import tarfile
import argparse
import glob
import sys
import time
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Tuple
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import xgboost as xgb

# s3fs (required only if using s3://)
try:
    import s3fs
    _HAS_S3FS = True
except Exception:
    _HAS_S3FS = False

# Matplotlib optional
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False


# ----------------------------
# Utils S3 / files
# ----------------------------

def _is_s3(path: str) -> bool:
    return str(path).startswith("s3://")


def _require_s3fs_if_needed(*paths: str):
    if any(_is_s3(p) for p in paths):
        if not _HAS_S3FS:
            raise RuntimeError("s3fs is required for s3:// paths. Install s3fs.")


def _fs():
    # assumes _require_s3fs_if_needed already checked
    return s3fs.S3FileSystem(anon=False)


def _read_text(path: str) -> str:
    if _is_s3(path):
        fs = _fs()
        with fs.open(path, "rb") as f:
            return f.read().decode("utf-8")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _read_json(path: str) -> Dict[str, Any]:
    return json.loads(_read_text(path))


def _write_bytes(path: str, data: bytes):
    if _is_s3(path):
        fs = _fs()
        with fs.open(path, "wb") as f:
            f.write(data)
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)


def _write_json(path: str, obj: Dict[str, Any]):
    _write_bytes(path, json.dumps(obj, indent=2, ensure_ascii=False).encode("utf-8"))


def _list_csv_parts(prefix: str) -> List[str]:
    """Lists part-*.csv under prefix (S3 or local), or returns [prefix] if prefix endswith .csv."""
    if prefix.endswith(".csv"):
        return [prefix]

    if _is_s3(prefix):
        fs = _fs()
        no_scheme = prefix.replace("s3://", "")
        candidates = fs.glob(no_scheme + "/part-*.csv")
        return [f"s3://{p}" for p in sorted(candidates)]

    pattern = os.path.join(prefix, "part-*.csv")
    return sorted(glob.glob(pattern))


# ----------------------------
# Batch loader (Stage3_GO)
# ----------------------------

def _load_csv_matrix(prefix: str, label_col: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stage3_GO:
      - CSVs without header
      - columns: [label] + features

    Returns:
      X: (n, n_features)
      y: (n,)
    """
    parts = _list_csv_parts(prefix)
    if not parts:
        raise RuntimeError(f"No CSV found under {prefix} (expected part-*.csv or a CSV file)")

    xs, ys = [], []
    for p in parts:
        if _is_s3(p):
            fs = _fs()
            with fs.open(p, "rb") as f:
                df = pd.read_csv(f, header=None)
        else:
            df = pd.read_csv(p, header=None)

        if df.shape[1] < 2:
            raise RuntimeError(f"CSV shape too small ([label]+features) for {p}: {df.shape}")

        arr = df.to_numpy()
        y = arr[:, 0].astype(np.float32, copy=False)
        X = arr[:, 1:].astype(np.float32, copy=False)
        ys.append(y)
        xs.append(X)

    X_all = np.vstack(xs).astype(np.float32, copy=False)
    y_all = np.concatenate(ys).astype(np.float32, copy=False)
    return X_all, y_all


# ----------------------------
# XGBoost model loader
# ----------------------------

def _load_booster_from_model_tar(model_tar_uri: str) -> xgb.Booster:
    """Loads a SageMaker model.tar.gz and returns an xgboost.Booster."""
    if _is_s3(model_tar_uri):
        fs = _fs()
        with fs.open(model_tar_uri, "rb") as f:
            raw = f.read()
    else:
        with open(model_tar_uri, "rb") as f:
            raw = f.read()

    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tar:
        member = None
        for m in tar.getmembers():
            if m.name.endswith(("model.json", "xgboost-model", "model.bin")):
                member = m
                break
        if member is None:
            raise RuntimeError("No model.json/xgboost-model/model.bin inside model.tar.gz")
        model_bytes = tar.extractfile(member).read()

    booster = xgb.Booster()
    try:
        booster.load_model(io.BytesIO(model_bytes))  # xgboost>=2.0
    except Exception:
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp.write(model_bytes)
            tmp.flush()
            booster.load_model(tmp.name)
    return booster


# ----------------------------
# Metrics (AUPRC)
# ----------------------------

def _auprc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    try:
        from sklearn.metrics import average_precision_score
        return float(average_precision_score(y_true, y_pred))
    except Exception:
        # fallback approximation
        order = np.argsort(-y_pred)
        y = y_true[order]
        tp = np.cumsum(y == 1)
        fp = np.cumsum(y == 0)
        prec = tp / np.maximum(tp + fp, 1)
        rec = tp / max(np.sum(y == 1), 1)
        return float(np.trapz(prec, rec))


# ----------------------------
# Manifest helpers
# ----------------------------

def _load_feat_order(manifest_obj: Dict[str, Any], fallback_cols_count: int) -> List[str]:
    dc = manifest_obj.get("data_contract", {}) or {}
    feats = dc.get("features")
    if isinstance(feats, list) and feats:
        return list(feats)
    return [f"f{i}" for i in range(fallback_cols_count)]


# ----------------------------
# Main
# ----------------------------

def main():
    import traceback

    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest-s3", required=True, help="s3://.../manifest.json or local path")
    ap.add_argument("--batch", required=True, help="S3/local dir with part-*.csv or a single CSV file")
    ap.add_argument("--threshold", type=float, default=None,
                    help="Override global proba threshold (default: manifest.inference.decision_threshold)")
    ap.add_argument("--out", default=None,
                    help="S3 or local prefix for outputs (report.json, proba_hist.png, predictions.csv)")
    ap.add_argument("--limit", type=int, default=0, help="Optional row cap for quick tests")
    args = ap.parse_args()

    t0 = time.time()
    print(f"[infer] manifest={args.manifest_s3}")

    try:
        # s3fs needed if any s3 path is used (manifest, batch, out)
        _require_s3fs_if_needed(args.manifest_s3, args.batch, args.out or "")

        manifest = _read_json(args.manifest_s3)

        model_uri = manifest["model_uri"]
        _require_s3fs_if_needed(model_uri)

        infer_cfg = manifest.get("inference", {}) or {}
        manifest_thr = float(infer_cfg.get("decision_threshold", 0.5))

        cli_thr = args.threshold
        use_thr = float(cli_thr if cli_thr is not None else manifest_thr)
        threshold_source = "cli" if cli_thr is not None else "manifest"

        print(f"[infer] model_uri={model_uri}")
        print(f"[infer] decision_threshold={use_thr:.6f} (source={threshold_source})")

        # --- Stage3_GO contract checks (soft) ---
        dc = manifest.get("data_contract", {}) or {}
        if dc.get("drop_cols"):
            raise RuntimeError("Stage3_GO: drop_cols doit être [].")
        if bool(dc.get("normalize_at_infer", False)):
            raise RuntimeError("Stage3_GO: normalize_at_infer doit être False (normalisation faite en amont ou non utilisée).")

        label_col = str(dc.get("label_col", "Y"))

        # --- load batch ---
        X_raw, y = _load_csv_matrix(args.batch, label_col=label_col)
        if args.limit and args.limit > 0:
            X_raw = X_raw[:args.limit]
            y = y[:args.limit]

        print(f"[infer] batch loaded: X_raw={X_raw.shape}, y=present")

        # --- feature order ---
        feat_names = _load_feat_order(manifest, fallback_cols_count=X_raw.shape[1])
        if len(feat_names) != X_raw.shape[1]:
            raise RuntimeError(
                f"[features-order] Mismatch: feat_names({len(feat_names)}) vs X_raw width({X_raw.shape[1]})"
            )

        X_df = pd.DataFrame(X_raw, columns=feat_names)
        X_df = X_df.loc[:, feat_names]  # enforce order
        X = X_df.astype(np.float32)

        nan_frac = float(np.isnan(X.values).mean())
        if nan_frac > 0:
            print(f"[warn] NaN in X: {nan_frac:.6f} frac")

        # --- model ---
        print("[infer] loading model booster from S3..." if _is_s3(model_uri) else "[infer] loading model booster...")
        booster = _load_booster_from_model_tar(model_uri)

        try:
            dmat = xgb.DMatrix(X.values, feature_names=list(feat_names))
        except Exception:
            dmat = xgb.DMatrix(X.values)

        # --- predict ---
        print("[infer] predicting probabilities (P(Y=1))...")
        probas = np.asarray(booster.predict(dmat), dtype=np.float32)

        q = np.quantile(probas, [0, 0.5, 0.9, 0.95, 0.99, 1.0])
        print(
            f"[debug] p_pred quantiles: min={q[0]:.4f} p50={q[1]:.4f} p90={q[2]:.4f} "
            f"p95={q[3]:.4f} p99={q[4]:.4f} max={q[5]:.4f}"
        )

        # --- decision ---
        yhat = (probas >= use_thr).astype(np.int32)
        print(f"[debug] count>=thr: {int((probas >= use_thr).sum())} / {int(probas.size)}")

        # --- metrics/report ---
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
        y_int = y.astype(np.int32)
        apv = _auprc(y_int, probas)

        tp = int(np.sum((y_int == 1) & (yhat == 1)))
        fp = int(np.sum((y_int == 0) & (yhat == 1)))
        tn = int(np.sum((y_int == 0) & (yhat == 0)))
        fn = int(np.sum((y_int == 1) & (yhat == 0)))
        precision = float(tp / max(tp + fp, 1))
        recall = float(tp / max(tp + fn, 1))
        f1 = float((2 * precision * recall) / max(precision + recall, 1e-12))
        trigger_rate = float(np.mean(yhat))
        pos_rate = float(np.mean(y_int))

        report: Dict[str, Any] = {
            "model_uri": model_uri,
            "timestamp": ts,
            "manifest": args.manifest_s3,
            "batch": args.batch,
            "decision_threshold_manifest": float(manifest_thr),
            "decision_threshold_used": float(use_thr),
            "threshold_source": threshold_source,
            "n_rows": int(len(probas)),
            "AUPRC_batch": float(apv),
            "precision_at_threshold": float(precision),
            "recall_at_threshold": float(recall),
            "F1_at_threshold": float(f1),
            "TP": tp, "FP": fp, "TN": tn, "FN": fn,
            "trigger_rate": float(trigger_rate),
            "pos_rate": float(pos_rate),
            "nan_frac_X": float(nan_frac),
        }

        print(
            f"[infer] AUPRC={apv:.6f} | P@thr={precision:.3f} R@thr={recall:.3f} "
            f"F1@thr={f1:.3f} | trigger_rate={trigger_rate:.4f} pos_rate={pos_rate:.4f}"
        )

        # --- outputs ---
        if args.out:
            out_prefix = args.out.rstrip("/")

            report_uri = out_prefix + "/report.json"
            _write_json(report_uri, report)
            print(f"[infer] report written: {report_uri}")

            pred_uri = out_prefix + "/predictions.csv"
            df_pred = pd.DataFrame({"y": y_int, "p_pred": probas, "yhat": yhat})
            if _is_s3(pred_uri):
                fs = _fs()
                with fs.open(pred_uri, "w") as f:
                    df_pred.to_csv(f, index=False)
            else:
                Path(pred_uri).parent.mkdir(parents=True, exist_ok=True)
                df_pred.to_csv(pred_uri, index=False)
            print(f"[infer] predictions written: {pred_uri}")

            if _HAS_MPL:
                try:
                    fig = plt.figure(figsize=(6, 4), dpi=120)
                    ax = fig.add_subplot(111)
                    ax.hist(probas, bins=50, alpha=0.85)
                    ax.set_title("Prediction probability histogram (P(Y=1))")
                    ax.set_xlabel("p(Y=1)")
                    ax.set_ylabel("count")
                    ax.axvline(use_thr, linestyle="--")
                    fig.tight_layout()
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png")
                    plt.close(fig)
                    buf.seek(0)
                    png_uri = out_prefix + "/proba_hist.png"
                    _write_bytes(png_uri, buf.getvalue())
                    print(f"[infer] histogram written: {png_uri}")
                except Exception as e:
                    print(f"[warn] could not write histogram: {e}")
            else:
                print("[infer] matplotlib not available; skipping histogram.")
        else:
            print("[infer] no --out provided; skipping artifact write.")

        print(f"[infer] done in {time.time() - t0:.1f}s")

    except Exception as e:
        print(f"[ERROR] {type(e).__name__}: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"[ERROR] {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)