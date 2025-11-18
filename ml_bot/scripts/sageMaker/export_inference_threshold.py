# ml_bot/scripts/sageMaker/export_inference_threshold.py
import argparse, json, sys, time
from urllib.parse import urlparse

import boto3
from botocore.exceptions import ClientError


def s3_read_json(s3, uri: str):
    """Lit un JSON depuis s3://... et le retourne sous forme de dict."""
    p = urlparse(uri, allow_fragments=False)
    if p.scheme != "s3":
        raise ValueError(f"URI S3 invalide: {uri}")
    obj = s3.get_object(Bucket=p.netloc, Key=p.path.lstrip("/"))
    return json.loads(obj["Body"].read().decode("utf-8"))


def s3_write_bytes(s3, uri: str, data: bytes, content_type: str = "text/plain"):
    p = urlparse(uri, allow_fragments=False)
    if p.scheme != "s3":
        raise ValueError(f"URI S3 invalide: {uri}")
    s3.put_object(Bucket=p.netloc, Key=p.path.lstrip("/"), Body=data, ContentType=content_type)


def sibling_uri(model_uri: str, filename: str) -> str:
    """Retourne s3://bucket/path/to/output/<filename> en partant de .../output/model.tar.gz."""
    p = urlparse(model_uri, allow_fragments=False)
    key = p.path.lstrip("/")
    if not key.endswith("model.tar.gz"):
        raise ValueError(f"model_uri inattendu (finit pas par model.tar.gz): {model_uri}")
    out_prefix = key[: -len("model.tar.gz")]  # garde ".../output/"
    return f"s3://{p.netloc}/{out_prefix}{filename}"


def main():
    ap = argparse.ArgumentParser(description="Export du seuil d'inférence depuis _summary.json vers S3")
    ap.add_argument("--summary-uri", required=True, help="s3://.../_summary.json issu de la grid")
    ap.add_argument("--region", default="ap-northeast-1")
    args = ap.parse_args()

    s3 = boto3.client("s3", region_name=args.region)

    try:
        summary = s3_read_json(s3, args.summary_uri)
    except ClientError as e:
        print(f"[ERR] Impossible de lire summary: {args.summary_uri} — {e}", file=sys.stderr)
        sys.exit(2)

    best = summary.get("best") or {}
    model_uri = best.get("model_uri")
    thr = best.get("best_threshold")

    if model_uri is None or thr is None:
        print("[ERR] Champs manquants dans summary.best: 'model_uri' et/ou 'best_threshold'", file=sys.stderr)
        sys.exit(3)

    try:
        thr = float(thr)
    except Exception:
        print(f"[ERR] best_threshold non convertible en float: {thr}", file=sys.stderr)
        sys.exit(4)

    if not (0.0 <= thr <= 1.0):
        print(f"[ERR] best_threshold hors [0,1]: {thr}", file=sys.stderr)
        sys.exit(5)

    thr_uri = sibling_uri(model_uri, "inference_threshold.txt")
    meta_uri = sibling_uri(model_uri, "deployed_meta.json")

    # 1) Écrit le seuil en clair
    s3_write_bytes(s3, thr_uri, f"{thr:.10f}\n".encode("utf-8"), content_type="text/plain")

    # 2) Écrit un petit méta pour l’intégration prod / traçabilité
    deployed = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_uri": model_uri,
        "inference_threshold": thr,
        "source_summary_uri": args.summary_uri,
        # infos utiles pour recharger la normalisation côté prod (optionnel si tu les as ailleurs)
        "data_root": summary.get("data_root"),
        "scaler_stats_uri_hint": f"{summary.get('data_root','')}/_meta/scaler_stats.json",
    }
    s3_write_bytes(s3, meta_uri, json.dumps(deployed, indent=2).encode("utf-8"), content_type="application/json")

    print("✅ Export OK")
    print("  model_uri:            ", model_uri)
    print("  inference_threshold:  ", thr)
    print("  threshold written to: ", thr_uri)
    print("  meta written to:      ", meta_uri)


if __name__ == "__main__":
    main()


# python ml_bot/scripts/sageMaker/export_inference_threshold.py \
#  --summary-uri s3://tradebot-config-tokyo/models/xgb/grid/20251106-173607_summary.json \
#  --region ap-northeast-1