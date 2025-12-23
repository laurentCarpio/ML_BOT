#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_ev_thresholds_per_pocket.py

À partir d'un fichier de prédictions (predictions.csv) contenant au moins :
  - symbol, tf, t, EV
  - idéalement aussi Y, p_pred, rr_nominal pour debug

Calibre :
  1) un seuil EV global (ev_min_global) pour atteindre ~target_trades_per_day
  2) des seuils EV par poche (symbol × tf) — option 1 "safe" :

     - si EV_p90 >= ev_min_global  → ev_min_pocket = ev_min_global
     - sinon si EV_p80 >= ev_min_global → ev_min_pocket = EV_p80
     - sinon → ev_min_pocket = max(ev_floor, EV_median * 1.5)

Sortie : JSON avec :
  - meta
  - global
  - pockets["SYMBOL|TF"] = { ev_min, stats... }

Exemple :

  python ml_bot/scripts/sageMaker/make_ev_thresholds_per_pocket.py \
  --predictions s3://tradebot-config-tokyo/tmp/pred_debug/2025-11-25T14-58-01Z/predictions.csv \
  --target-trades-per-day 7.5 \
  --ev-floor 0.0 \
  --out-json s3://tradebot-config-tokyo/models/xgb/infer_runs/v51-short/ev_thresholds_per_pocket.json
"""

import argparse
import json
import sys
from typing import Dict, Any

import numpy as np
import pandas as pd

# s3fs est requis pour lire/écrire sur S3 via pandas
try:
    import s3fs  # noqa: F401
    _HAS_S3FS = True
except Exception:
    _HAS_S3FS = False


def _is_s3(path: str) -> bool:
    return str(path).startswith("s3://")


def _write_json(path: str, obj: Dict[str, Any]) -> None:
    data = json.dumps(obj, indent=2, ensure_ascii=False).encode("utf-8")
    if _is_s3(path):
        if not _HAS_S3FS:
            raise RuntimeError("s3fs est requis pour écrire sur S3.")
        fs = s3fs.S3FileSystem(anon=False)
        with fs.open(path, "wb") as f:
            f.write(data)
    else:
        import os
        from pathlib import Path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)


def _infer_days(df: pd.DataFrame) -> int:
    if "t" not in df.columns:
        raise RuntimeError("Colonne 't' absente : impossible d'inférer le nombre de jours.")
    # t peut être string → to_datetime + .dt.date
    t = pd.to_datetime(df["t"], errors="coerce", utc=True)
    n_days = t.dt.date.nunique()
    if n_days <= 0:
        raise RuntimeError("Impossible de calculer un nombre de jours > 0 depuis 't'.")
    print(f"[time] nombre de jours uniques dans le set = {n_days}")
    return int(n_days)


def _pick_global_ev_min(ev: np.ndarray,
                        target_trades_per_day: float,
                        n_days: int,
                        ev_floor: float = 0.0) -> float:
    """
    Calibre un EV minimal global pour atteindre target_trades_per_day * n_days
    sur la distribution EV (en filtrant EV >= ev_floor).
    """
    ev = np.asarray(ev, dtype=float)
    ev = ev[np.isfinite(ev)]
    ev = ev[ev >= ev_floor]

    if ev.size == 0:
        raise RuntimeError("Aucun EV valide (>= ev_floor) pour calibrer le seuil global.")

    n_target = int(round(target_trades_per_day * n_days))
    n_target = max(1, n_target)  # au moins 1 trade sur tout le set

    # tri décroissant
    ev_sorted = np.sort(ev)[::-1]

    if n_target >= ev_sorted.size:
        # pas assez de points → on prend le min
        ev_min = float(ev_sorted[-1])
    else:
        ev_min = float(ev_sorted[n_target - 1])

    # log info
    triggered = int((ev >= ev_min).sum())
    trades_per_day = triggered / float(n_days)

    print("\n[calibration EV] global")
    print(f"  n_days_eff            = {n_days}")
    print(f"  target_trades_total   = {target_trades_per_day * n_days:.1f}")
    print(f"  ev_min_global choisi  = {ev_min:.6f}")
    print(f"  trades déclenchés     = {triggered}")
    print(f"  trades/jour réalisés  = {trades_per_day:.2f}")

    return ev_min


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True,
                    help="CSV des prédictions (S3 ou local). Doit contenir au moins: symbol, tf, t, EV.")
    ap.add_argument("--target-trades-per-day", type=float, default=7.5,
                    help="Objectif global de trades par jour (ex: 7.5).")
    ap.add_argument("--ev-floor", type=float, default=0.0,
                    help="EV minimal absolu pour être considéré dans le calibrage global et par poche.")
    ap.add_argument("--out-json", required=True,
                    help="Chemin S3/local pour écrire le JSON de seuils EV par poche.")
    args = ap.parse_args()

    print(f"[cfg] predictions = {args.predictions}")
    print(f"[cfg] target_trades_per_day = {args.target_trades_per_day}")
    print(f"[cfg] ev_floor = {args.ev_floor}")
    print(f"[cfg] out_json = {args.out_json}")

    # --- 1) load predictions ---
    try:
        df = pd.read_csv(args.predictions)
    except Exception as e:
        print(f"[ERROR] Impossible de lire {args.predictions} : {e}", file=sys.stderr)
        sys.exit(1)

    print(f"[info] loaded predictions: shape={df.shape}")
    print(f"[cols] {list(df.columns)}")

    required_cols = {"symbol", "tf", "t", "EV"}
    missing = required_cols - set(df.columns)
    if missing:
        raise RuntimeError(f"Colonnes manquantes dans predictions.csv: {sorted(missing)}")

    # --- 2) EV distribution globale pour info ---
    ev_series = pd.to_numeric(df["EV"], errors="coerce")
    print("\n[EV] distribution globale :")
    print(ev_series.describe(percentiles=[0.5, 0.8, 0.9, 0.95, 0.99]))

    # --- 3) nb de jours ---
    n_days = _infer_days(df)

    # --- 4) calibrage global ev_min ---
    ev_min_global = _pick_global_ev_min(
        ev=ev_series.values,
        target_trades_per_day=args.target_trades_per_day,
        n_days=n_days,
        ev_floor=args.ev_floor,
    )

    # recompte avec ce seuil global (en filtrant EV >= ev_floor aussi)
    ev_valid = ev_series.copy()
    ev_valid = ev_valid[(ev_valid >= args.ev_floor) & ev_valid.notna()]
    n_trig_global = int((ev_valid >= ev_min_global).sum())
    trades_per_day_global = n_trig_global / float(n_days)

    # --- 5) stats par pocket symbol × tf (option 1 safe) ---
    pockets = {}
    print("\n[per symbol/tf] stats EV et seuils par poche (option 1 safe):")

    # groupby sur (symbol, tf)
    grp = df.groupby(["symbol", "tf"], sort=True)

    header_printed = False
    rows_for_print = []

    for (symbol, tf), g in grp:
        ev = pd.to_numeric(g["EV"], errors="coerce")
        ev = ev[ev.notna()]

        if ev.empty:
            continue

        # stats de base
        ev_mean = float(ev.mean())
        ev_p50 = float(ev.quantile(0.50))
        ev_p80 = float(ev.quantile(0.80))
        ev_p90 = float(ev.quantile(0.90))

        # nombre de triggers avec le seuil global
        trig_global = int((ev >= ev_min_global).sum())

        # --- règle option 1 (safe) ---
        if ev_p90 >= ev_min_global:
            ev_min_p = ev_min_global
            rule = "p90>=global → global"
        elif ev_p80 >= ev_min_global:
            ev_min_p = ev_p80
            rule = "p80>=global → p80"
        else:
            ev_min_p = max(args.ev_floor, ev_p50 * 1.5)
            rule = "else → 1.5×median (floor)"

        # triggers avec le seuil de poche
        trig_pocket = int((ev >= ev_min_p).sum())
        trades_per_day_pocket = trig_pocket / float(n_days)

        key = f"{symbol}|{tf}"
        pockets[key] = {
            "symbol": symbol,
            "tf": tf,
            "rows": int(g.shape[0]),
            "ev_mean": ev_mean,
            "ev_p50": ev_p50,
            "ev_p80": ev_p80,
            "ev_p90": ev_p90,
            "ev_min": float(ev_min_p),
            "rule": rule,
            "triggers_global": int(trig_global),
            "triggers_pocket": int(trig_pocket),
            "trades_per_day_pocket": float(trades_per_day_pocket),
        }

        rows_for_print.append(
            (symbol, tf, g.shape[0], ev_mean, ev_p50, ev_p80, ev_p90,
             ev_min_p, trig_global, trig_pocket, trades_per_day_pocket, rule)
        )

    # tri pour affichage (par triggers_pocket décroissant)
    rows_for_print.sort(key=lambda r: r[9], reverse=True)

    print("  symbol   tf  rows  ev_mean   ev_p50   ev_p80   ev_p90  ev_min  "
          "trig_global  trig_pocket  trades_per_day  rule")
    for (symbol, tf, rows, ev_mean, ev_p50, ev_p80, ev_p90,
         ev_min_p, trig_global, trig_pocket, tpd, rule) in rows_for_print:
        print(f"{symbol:>8} {tf:>4} "
              f"{rows:5d} "
              f"{ev_mean:8.6f} {ev_p50:8.6f} {ev_p80:8.6f} {ev_p90:8.6f} "
              f"{ev_min_p:8.6f} "
              f"{trig_global:12d} {trig_pocket:12d} {tpd:14.6f}  {rule}")

    # --- 6) JSON final ---
    out_obj: Dict[str, Any] = {
        "meta": {
            "n_rows": int(df.shape[0]),
            "n_days": int(n_days),
            "target_trades_per_day": float(args.target_trades_per_day),
            "ev_floor": float(args.ev_floor),
        },
        "global": {
            "ev_min": float(ev_min_global),
            "trades_total": int(n_trig_global),
            "trades_per_day": float(trades_per_day_global),
        },
        "pockets": pockets,
    }

    _write_json(args.out_json, out_obj)
    print(f"\n[ok] EV per-pocket thresholds JSON written to: {args.out_json}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"[ERROR] {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)