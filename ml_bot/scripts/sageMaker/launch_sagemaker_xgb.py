# launch_sagemaker_xgb.py
import os, time, json, boto3
from pathlib import Path
from sagemaker import Session
from sagemaker.inputs import TrainingInput
from sagemaker.xgboost import XGBoost

# === À ADAPTER : ===
REGION    = boto3.Session().region_name or "ap-northeast-1"
ROLE_ARN  = "arn:aws:iam::174175447862:role/AmazonSageMaker-ExecutionRole"
ROOT_S3   = "s3://tradebot-config-tokyo/data/stage3-xgb/v1-go"
MODEL_S3  = "s3://tradebot-config-tokyo/models/xgb"  # dossier où SageMaker déposera model.tar.gz
MODEL_SID = MODEL_S3

sess = Session()
image_version = "1.7-1"  # conteneur XGBoost officiel (script mode OK)

# charge scale_pos_weight depuis _meta si présent
posw = 1.0
s3 = boto3.client("s3", region_name=REGION)
try:
    from urllib.parse import urlparse
    meta_url = urlparse(ROOT_S3 + "/_meta/train_pos_weight.json")
    obj = s3.get_object(Bucket=meta_url.netloc, Key=meta_url.path.lstrip("/"))
    posw = float(json.loads(obj["Body"].read())["pos_weight"])
    print(f"[meta] scale_pos_weight = {posw:.4f}")
except Exception as e:
    print("[meta] pos_weight introuvable, on continue:", e)

SRC_DIR  = Path(__file__).parent        # ml_bot/scripts/sageMaker
ENTRY    = "train.py"                    # doit exister dans SRC_DIR

est = XGBoost(
    entry_point=ENTRY,           # notre script ci-dessous
    source_dir=str(SRC_DIR),                   # dossier courrant (contient train.py)
    role=ROLE_ARN,
    instance_type="ml.c5.xlarge",
    instance_count=1,
    framework_version=image_version,
    py_version="py3",
    hyperparameters={
        # hyperparams XGBoost
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "tree_method": "hist",
        "eta": 0.05,
        "max_depth": 8,
        "min_child_weight": 1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "num_round": 800,
        "early_stopping_rounds": 100,
        # on passe scale_pos_weight au script (peut être surchargé dans train.py si _meta diffère)
        "scale_pos_weight": posw,
    },

    output_path=MODEL_SID,
    use_spot_instances=True,
    max_run=3600,
    max_wait=7200,
)

channels = {
    # Ces noms deviennent des dossiers dans le conteneur: $SM_CHANNEL_TRAIN, etc.
    "train":              TrainingInput(s3_data=f"{ROOT_S3}/train",       content_type="text/csv"),
    "validation":         TrainingInput(s3_data=f"{ROOT_S3}/validation",  content_type="text/csv"),
    "train_weight":       TrainingInput(s3_data=f"{ROOT_S3}/train/train_weight", content_type="text/csv"),
    "validation_weight":  TrainingInput(s3_data=f"{ROOT_S3}/validation/validation_weight", content_type="text/csv"),
    "meta":               TrainingInput(s3_data=f"{ROOT_S3}/_meta",       content_type="application/x-directory"),
}

job_name = f"xgb-go-{time.strftime('%Y%m%d-%H%M%S')}"
print("Submitting job:", job_name)
est.fit(channels, job_name=job_name, logs=True)

print("✅ Done. Model at:", est.model_data)
print("Saved to:", MODEL_SID)