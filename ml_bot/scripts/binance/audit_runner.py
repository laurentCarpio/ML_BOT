#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import yaml
import zipfile
from io import TextIOWrapper

import pandas as pd
import s3fs
import fsspec

from ml_bot.scripts.binance.sanity_reader import apply_sanity
from ml_bot.scripts.binance.anomaly_scan import scan_anomalies


def load_binance_1m_from_s3(symbol: str,
                            interval: str,
                            s3_daily_root: str) -> pd.DataFrame:
    """
    Lit tous les ZIP journaliers 1m Binance pour un symbole sur S3
    et renvoie un DataFrame concaténé, trié par timestamp.

    Entrée attendue :
      s3_daily_root/<SYMBOL>/<interval>/<SYMBOL>-<interval>-YYYY-MM-DD.zip

    Colonnes :
      - renomme open_time -> timestamp (datetime, tz-naive)
      - garde toutes les autres colonnes telles quelles
    """
    fs = s3fs.S3FileSystem(anon=False)

    prefix = f"{s3_daily_root.rstrip('/')}/{symbol}/{interval}/"
    print(f"[info] Listing ZIP under: {prefix}")

    zip_paths = sorted(fs.glob(prefix + f"{symbol}-{interval}-*.zip"))
    if not zip_paths:
        print(f"[warn] Aucun ZIP trouvé pour {symbol} @ {interval}")
        return pd.DataFrame()

    dfs = []

    for zip_path in zip_paths:
        try:
            print(f"[info] Lecture ZIP: {zip_path}")
            with fs.open(zip_path, "rb") as fobj:
                with zipfile.ZipFile(fobj, "r") as zf:
                    inner_names = [n for n in zf.namelist()
                                   if n.endswith(".csv")] or zf.namelist()
                    inner_file = inner_names[0]

                    with zf.open(inner_file) as inner_f:
                        df = pd.read_csv(
                            TextIOWrapper(inner_f, encoding="utf-8"),
                            header=0,
                        )

            if "open_time" not in df.columns:
                raise ValueError(f"Colonne 'open_time' absente dans {zip_path}")

            df = df.dropna()
            if df.empty:
                print(f"[warn] DF vide pour {zip_path}, ignoré")
                continue

            # open_time (ms) -> timestamp (datetime)
            df.rename(columns={"open_time": "timestamp"}, inplace=True)
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

            dfs.append(df)

        except Exception as e:
            print(f"❌ Erreur lecture ZIP {zip_path} : {e}")

    if not dfs:
        print(f"[warn] Aucun DF valide pour {symbol}")
        return pd.DataFrame()

    df_full = pd.concat(dfs, ignore_index=True)
    df_full = df_full.dropna().sort_values("timestamp").reset_index(drop=True)
    return df_full


def consecutive_runs(ts_series: pd.Series):
    """
    ts_series: Series de timestamps (string/datetime), non nécessairement triée.
    Retourne la liste de tuples (start, end) pour les runs consécutifs à +60s.
    """
    s = pd.to_datetime(ts_series).sort_values().reset_index(drop=True)
    if s.empty:
        return []
    runs = []
    start = s.iloc[0]
    prev = s.iloc[0]
    for t in s.iloc[1:]:
        if (t - prev).total_seconds() == 60:
            prev = t
        else:
            runs.append((start, prev))
            start = t
            prev = t
    runs.append((start, prev))
    return runs


def write_json(path: str, data):
    """
    Ecrit un JSON sur path, qui peut être local ou s3://...
    """
    with fsspec.open(path, "w") as f:
        json.dump(data, f, default=str, indent=2)


def write_csv(path: str, df: pd.DataFrame):
    """
    Ecrit un CSV sur path, local ou s3://...
    """
    with fsspec.open(path, "w") as f:
        df.to_csv(f, index=False)


def write_parquet(path: str, df: pd.DataFrame):
    """
    Ecrit un Parquet sur path, local ou s3://...
    """
    with fsspec.open(path, "wb") as f:
        df.to_parquet(f, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # === Config S3 avec défauts adaptés à ton setup ===
    s3_cfg = cfg.get("s3", {})
    s3_daily_root = s3_cfg.get(
        "binance_daily_root",
        "s3://tradebot-config-tokyo/data/binance/futures/um/daily/klines",
    )
    s3_bougies_root = s3_cfg.get(
        "bougies_root",
        "s3://tradebot-config-tokyo/data/bougies",
    )

    report_dir = cfg["outputs"]["report_dir"]
    # Report_dir peut être local ou s3://...
    # Si local, on crée le dossier
    if not report_dir.startswith("s3://"):
        os.makedirs(report_dir, exist_ok=True)

    summary = []

    # On parcourt les entrées déclarées (une par symbole)
    for item in cfg["data"]["inputs"]:
        symbol = item["symbol"]
        freq = item.get("freq", "1m")

        print("=" * 80)
        print(f"[symbol] {symbol} @ {freq}")
        print("=" * 80)

        # 1) Charger TOUTE l'historique depuis les ZIP journaliers sur S3
        df_full = load_binance_1m_from_s3(
            symbol=symbol,
            interval=freq,
            s3_daily_root=s3_daily_root,
        )

        if df_full.empty:
            print(f"⛔ Aucun data trouvé pour {symbol}, skip.")
            continue

        # 2) Sanity + anomalies (apply_sanity accepte df_in)
        df_clean, sanity = apply_sanity(cfg, symbol, df_in=df_full)
        anomalies = scan_anomalies(df_clean, cfg)

        # 3) Construire le set des timestamps à retirer
        to_drop = set()

        # a) zero_volume : drop tout
        zero_evt = anomalies[anomalies["type"] == "zero_volume"]["timestamp"]
        to_drop.update(pd.to_datetime(zero_evt).tolist())

        # b) flat_candle : drop seulement les runs >= 5 minutes
        flat_evt = anomalies[anomalies["type"] == "flat_candle"]["timestamp"]
        flat_runs = consecutive_runs(flat_evt)
        for a, b in flat_runs:
            run_len = int((b - a).total_seconds() / 60) + 1
            if run_len >= 5:
                t = a
                while t <= b:
                    to_drop.add(t)
                    t = t + pd.Timedelta(minutes=1)

        # 4) Appliquer le drop sur df_clean
        df_clean["timestamp"] = pd.to_datetime(df_clean["timestamp"])
        mask_bad = df_clean["timestamp"].isin(list(to_drop))
        nb_before = len(df_clean)
        df_clean = df_clean.loc[~mask_bad].copy()
        nb_after = len(df_clean)
        print(f"🧹 Filtrage auto: retiré {nb_before - nb_after} lignes (zero_volume/flat runs)")

        # 5) Ecrire les rapports (sanity + anomalies) dans report_dir
        base_report = report_dir.rstrip("/")

        sanity_path = f"{base_report}/{symbol}_sanity.json"
        anomalies_path = f"{base_report}/{symbol}_anomalies.csv"

        write_json(sanity_path, sanity)
        write_csv(anomalies_path, anomalies)
        print(f"💾 Sanity JSON : {sanity_path}")
        print(f"💾 Anomalies CSV : {anomalies_path}")

        # 6) Sauvegarde la série nettoyée en Parquet annuel sur S3
        df_clean = df_clean.sort_values("timestamp").reset_index(drop=True)
        df_clean["year"] = df_clean["timestamp"].dt.year

        rows_total = 0
        for year, df_year in df_clean.groupby("year"):
            df_year = df_year.drop(columns=["year"]).copy()

            out_path = (
                f"{s3_bougies_root.rstrip('/')}/{symbol}/"
                f"{symbol}-{freq}-{year}.parquet"
            )

            write_parquet(out_path, df_year)
            rows_total += len(df_year)
            print(f"✅ Parquet {year} : {out_path} ({len(df_year)} lignes)")

        summary.append({
            "symbol": symbol,
            "rows": int(rows_total),
            "gaps": int(sanity.get("gaps_count", 0)),
            "duplicates": int(sanity.get("duplicates_count", 0)),
            "anomalies": int(len(anomalies)),
            "freq": freq,
        })

    # 7) SUMMARY global
    summary_path = f"{report_dir.rstrip('/')}/SUMMARY.json"
    write_json(summary_path, summary)
    print(f"📊 SUMMARY : {summary_path}")


if __name__ == "__main__":
    # Exemple d'appel :
    # python -u -m next_bot.core.backtest.audit_runner --config ./next_bot/core/backtest/config.yaml
    main()