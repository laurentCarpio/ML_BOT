#!/usr/bin/env python3
import argparse
import io
import time
from datetime import datetime, timedelta

import boto3
import requests

########################################################
#   Télécharger des k-lines Binance Futures UM
#   et les ENVOYER DIRECTEMENT vers S3 (sans stockage local)
########################################################

BASE_URL = "https://data.binance.vision/data/futures/um/daily/klines"
S3 = boto3.client("s3")


def _normalize_bucket_and_prefix(s3_bucket: str, s3_prefix: str):
    """
    Accepte:
      - s3_bucket="tradebot-config-tokyo"
      - s3_bucket="s3://tradebot-config-tokyo"
    et aussi si jamais tu fournis s3_prefix="s3://bucket/prefix/..."
    (auquel cas on découpe proprement).
    """
    bucket = s3_bucket.strip()
    prefix = s3_prefix.strip() if s3_prefix else ""

    # Cas où l'utilisateur met tout dans s3_prefix par erreur: s3://bucket/prefix/...
    if prefix.startswith("s3://"):
        # Exemple: s3://my-bucket/data/binance/...
        without = prefix[len("s3://"):]
        parts = without.split("/", 1)
        bucket = parts[0]
        prefix = parts[1] if len(parts) > 1 else ""

    # Cas simple: s3_bucket commence par s3://
    if bucket.startswith("s3://"):
        bucket = bucket[len("s3://"):]

    # On enlève les / superflus
    prefix = prefix.lstrip("/")

    return bucket, prefix


def download_and_upload_klines(
    symbol: str,
    interval: str,
    start_date: str,
    end_date: str,
    s3_bucket: str,
    s3_prefix: str,
    sleep_sec: float = 0.2,
):
    """
    Télécharge les ZIP kline Binance futures UM
    et upload directement vers S3 sans stockage local.
    """
    bucket, base_prefix = _normalize_bucket_and_prefix(s3_bucket, s3_prefix)

    cur = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    sess = requests.Session()
    sess.headers["User-Agent"] = "binance-klines-s3/1.0"

    while cur <= end:
        date_str = cur.strftime("%Y-%m-%d")
        filename = f"{symbol}-{interval}-{date_str}.zip"
        url = f"{BASE_URL}/{symbol}/{interval}/{filename}"

        # Chemin S3
        if base_prefix:
            s3_key = f"{base_prefix.rstrip('/')}/{symbol}/{interval}/{filename}"
        else:
            s3_key = f"{symbol}/{interval}/{filename}"

        # Vérifier si déjà dans S3
        try:
            S3.head_object(Bucket=bucket, Key=s3_key)
            print(f"✅ Déjà présent dans S3 : s3://{bucket}/{s3_key}")
            cur += timedelta(days=1)
            continue
        except S3.exceptions.ClientError:
            # Objet absent → on continue pour télécharger
            pass

        # HEAD pour savoir si le fichier existe chez Binance
        head = sess.head(url)
        if head.status_code == 404:
            print(f"❌ Introuvable (404) : {filename}")
            cur += timedelta(days=1)
            continue
        elif not head.ok:
            print(f"⚠️ HTTP {head.status_code} : {filename}")
            cur += timedelta(days=1)
            continue

        print(f"Téléchargement → {filename}")
        r = sess.get(url, stream=True)
        r.raise_for_status()

        # Lire tout en mémoire
        buffer = io.BytesIO()
        for chunk in r.iter_content(chunk_size=1 << 20):
            if chunk:
                buffer.write(chunk)
        buffer.seek(0)

        # Upload S3 direct
        print(f"⬆️ Upload vers s3://{bucket}/{s3_key}")
        S3.upload_fileobj(buffer, bucket, s3_key)

        # Pause anti-rate-limit
        time.sleep(sleep_sec)
        cur += timedelta(days=1)


########################################################
#   MAIN
########################################################

def main():
    parser = argparse.ArgumentParser(
        description="Télécharge des k-lines Binance Futures UM vers S3"
    )

    parser.add_argument(
        "--symbols", nargs="+", required=True,
        help="Liste de symboles ex: BTCUSDT ETHUSDT"
    )
    parser.add_argument(
        "--interval", required=True,
        help="Intervalle ex: 1m, 5m, 15m, 1h"
    )
    parser.add_argument(
        "--start", required=True,
        help="Date début YYYY-MM-DD"
    )
    parser.add_argument(
        "--end", required=True,
        help="Date fin YYYY-MM-DD"
    )
    parser.add_argument(
        "--s3-bucket", required=True,
        help='Bucket S3 de destination (ex: "tradebot-config-tokyo" ou "s3://tradebot-config-tokyo")'
    )
    parser.add_argument(
        "--s3-prefix", required=True,
        help='Préfixe S3 ex: "data/binance/futures/um/daily/klines" ou "s3://bucket/prefix"'
    )
    parser.add_argument(
        "--sleep", type=float, default=0.2,
        help="Pause entre fichiers (default 0.2 sec)"
    )

    args = parser.parse_args()

    for sym in args.symbols:
        print(f"\n=== Symbol {sym} ===")
        download_and_upload_klines(
            symbol=sym,
            interval=args.interval,
            start_date=args.start,
            end_date=args.end,
            s3_bucket=args.s3_bucket,
            s3_prefix=args.s3_prefix,
            sleep_sec=args.sleep,
        )


if __name__ == "__main__":
    main()