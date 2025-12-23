#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_ev_thresholds_from_predictions.py

À partir d'un fichier predictions.csv (sorti par infer_xgb.py avec policy='ev'),
calcule un seuil EV global pour viser ~N trades/jour, affiche des stats par
(symbol, tf) et écrit un JSON de seuils EV.

Exemple d'usage :

  python ml_bot/scripts/sageMaker/make_ev_thresholds_from_predictions.py \
  --predictions s3://tradebot-config-tokyo/tmp/pred_debug/2025-11-25T14-58-01Z/predictions.csv \
  --target-trades-per-day 7.5 \
  --out-json s3://tradebot-config-tokyo/models/xgb/infer_runs/v51-short/ev_thresholds.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd

# s3fs pour S3
try:
    import s3fs
    _HAS_S3FS = True
except Exception:
    _HAS_S3FS = False


# ----------------------------
# Helpers S3 / fichiers
# ----------------------------

def _is_s3(path: str) -> bool:
    return str(path).startswith("s3://")


def _fs():
    if not _HAS_S3FS:
        raise RuntimeError("s3fs is required for S3 paths. Install s3fs.")
    return s3fs.S3FileSystem(anon=False)


def _read_csv(path: str) -> pd.DataFrame:
    if _is_s3(path):
        fs = _fs()
        with fs.open(path, "rb") as f:
            return pd.read_csv(f)
    else:
        return pd.read_csv(path)


def _write_json(path: str, obj: Dict[str, Any]):
    data = json.dumps(obj, indent=2, ensure_ascii=False)
    if _is_s3(path):
        fs = _fs()
        with fs.open(path, "wb") as f:
            f.write(data.encode("utf-8"))
    else:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(data, encoding="utf-8")


# ----------------------------
# Core logic
# ----------------------------

def _compute_n_days(df: pd.DataFrame) -> int:
    """Déduit le nombre de jours uniques à partir de la colonne 't'."""
    if "t" not in df.columns:
        raise RuntimeError("Column 't' is required in predictions.csv for day counting.")
    ts = pd.to_datetime(df["t"], errors="coerce", utc=True)
    days = ts.dt.date.dropna().unique()
    return int(len(days))


def _pick_ev_global(df_ev: pd.Series,
                    n_days: int,
                    target_trades_per_day: float,
                    ev_floor: float = 0.0) -> Tuple[float, int, float]:
    """
    Choisit un ev_min_global pour viser target_trades_per_day.
    Retourne (ev_min_global, n_trades, trades_per_day).
    """
    df_ev = df_ev.astype(float)
    df_ev = df_ev.replace([np.inf, -np.inf], np.nan)
    df_ev = df_ev.dropna()

    if df_ev.empty:
        raise RuntimeError("No finite EV values found in predictions.")

    # tri décroissant
    ev_sorted = np.sort(df_ev.values)[::-1]

    target_total = int(round(target_trades_per_day * n_days))
    target_total = max(target_total, 1)

    if target_total >= ev_sorted.size:
        thr = float(max(ev_sorted.min(), ev_floor))
        n_trades = int((ev_sorted >= thr).sum())
    else:
        thr_raw = float(ev_sorted[target_total - 1])
        thr = max(thr_raw, ev_floor)
        n_trades = int((ev_sorted >= thr).sum())

    trades_per_day = n_trades / float(max(n_days, 1))

    return thr, n_trades, trades_per_day


def _build_pocket_stats(df: pd.DataFrame,
                        ev_min_global: float,
                        n_days: int) -> pd.DataFrame:
    """
    Construit un tableau par (symbol, tf) avec stats EV et nb de triggers (EV >= ev_min_global).
    """
    required_cols = {"symbol", "tf", "EV"}
    missing = required_cols - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing required columns in predictions.csv: {sorted(missing)}")

    g = (
        df.groupby(["symbol", "tf"], dropna=False)["EV"]
          .agg(["count", "mean", "median"])
          .rename(columns={"count": "rows", "mean": "ev_mean", "median": "ev_p50"})
    )

    # quantiles
    q80 = df.groupby(["symbol", "tf"])["EV"].quantile(0.80)
    q90 = df.groupby(["symbol", "tf"])["EV"].quantile(0.90)

    g["ev_p80"] = q80
    g["ev_p90"] = q90

    # triggers avec le seuil global
    triggers = (
        df.assign(trigger=(df["EV"] >= ev_min_global).astype(int))
          .groupby(["symbol", "tf"])["trigger"]
          .sum()
          .rename("triggers")
    )

    g = g.join(triggers, how="left").fillna({"triggers": 0})
    g["triggers"] = g["triggers"].astype(int)

    # trades/jour pour info
    g["trades_per_day"] = g["triggers"] / float(max(n_days, 1))

    # tri par nb de triggers décroissant
    g = g.sort_values(["triggers", "symbol", "tf"], ascending=[False, True, True])

    return g.reset_index()


def main():
    ap = argparse.ArgumentParser(
        description="Calibrer un seuil EV global (et par poche) à partir de predictions.csv."
    )
    ap.add_argument("--predictions", required=True,
                    help="Chemin S3 ou local vers predictions.csv (sorti par infer_xgb.py).")
    ap.add_argument("--target-trades-per-day", type=float, default=7.5,
                    help="Nombre de trades/jour visé (global). Défaut=7.5.")
    ap.add_argument("--ev-floor", type=float, default=0.0,
                    help="Plancher pour ev_min_global (défaut=0.0).")
    ap.add_argument("--out-json", required=True,
                    help="Chemin S3 ou local pour écrire le JSON de seuils EV.")
    ap.add_argument("--top-k-preview", type=int, default=30,
                    help="Nombre de lignes (symbol, tf) à afficher en preview.")
    args = ap.parse_args()

    print(f"[cfg] predictions = {args.predictions}")
    print(f"[cfg] target_trades_per_day = {args.target_trades_per_day}")
    print(f"[cfg] ev_floor = {args.ev_floor}")
    print(f"[cfg] out_json = {args.out_json}")

    # --- Charge predictions.csv ---
    df = _read_csv(args.predictions)
    print(f"[info] loaded predictions: shape={df.shape}")
    print(f"[cols] {list(df.columns)}")

    if "EV" not in df.columns:
        raise RuntimeError("Column 'EV' is required in predictions.csv (run infer_xgb.py with policy='ev').")

    # Nettoyage EV
    df["EV"] = pd.to_numeric(df["EV"], errors="coerce")
    df = df.replace({"EV": {np.inf: np.nan, -np.inf: np.nan}})
    n_before = len(df)
    df = df.dropna(subset=["EV"])
    n_after = len(df)
    if n_after < n_before:
        print(f"[warn] dropped {n_before - n_after} rows with non-finite EV.")

    # --- Distribution globale EV ---
    print("\n[EV] distribution globale :")
    print(df["EV"].describe(percentiles=[0.5, 0.8, 0.9, 0.95, 0.99]))

    # --- Nombre de jours ---
    n_days = _compute_n_days(df)
    print(f"\n[time] nombre de jours uniques dans le test/val set = {n_days}")

    # --- Choix du ev_min_global ---
    ev_min_global, n_trades, trades_per_day = _pick_ev_global(
        df["EV"],
        n_days=n_days,
        target_trades_per_day=args.target_trades_per_day,
        ev_floor=args.ev_floor,
    )

    print("\n[calibration EV] global")
    print(f"  n_days_eff            = {n_days}")
    print(f"  target_trades_total   = {args.target_trades_per_day * n_days:.1f}")
    print(f"  ev_min_global choisi  = {ev_min_global:.6f}")
    print(f"  trades déclenchés     = {n_trades}")
    print(f"  trades/jour réalisés  = {trades_per_day:.2f}")

    # --- Stats par poche (symbol, tf) ---
    pocket_stats = _build_pocket_stats(df, ev_min_global=ev_min_global, n_days=n_days)

    print("\n[per symbol/tf] stats EV et nb de triggers (EV >= ev_min_global):")
    with pd.option_context("display.max_columns", None, "display.width", 160):
        print(pocket_stats.head(args.top_k_preview).to_string(index=False))

    # --- JSON de seuils EV ---
    # Pour l'instant : même ev_min_global pour toutes les poches,
    # mais on structure déjà le JSON pour pouvoir spécialiser plus tard.
    ev_min_per_pocket: Dict[str, float] = {}
    for _, row in pocket_stats.iterrows():
        sym = str(row["symbol"])
        tf = str(row["tf"])
        key = f"{sym}|{tf}"
        ev_min_per_pocket[key] = float(ev_min_global)

    out_obj: Dict[str, Any] = {
        "meta": {
            "source_predictions": args.predictions,
            "n_rows": int(len(df)),
            "n_days": int(n_days),
            "target_trades_per_day": float(args.target_trades_per_day),
            "ev_floor": float(args.ev_floor),
        },
        "ev_min_global": float(ev_min_global),
        "ev_min_per_pocket": ev_min_per_pocket,
        "pocket_stats": pocket_stats.to_dict(orient="records"),
    }

    _write_json(args.out_json, out_obj)
    print(f"\n[ok] EV thresholds JSON written to: {args.out_json}")


if __name__ == "__main__":
    main()