# ml_bot/scripts/sageMaker/misc.py
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score

# --- S3 paths (keep the s3:// prefix!) ---
ROOT = "s3://tradebot-config-tokyo/data/stage3-xgb/v1-go"
val_csv   = f"{ROOT}/validation/part-00000.csv"
w_val_csv = f"{ROOT}/validation/validation_weight/part-00000.csv"

# If you saved the model.json in S3 too, point to it:
MODEL_URI = "s3://tradebot-config-tokyo/models/xgb/xgb-go-20251106-141046/output/model.tar.gz"

# --- Read validation set from S3 ---
# Requires s3fs + aiobotocore installed and AWS creds configured.
val = pd.read_csv(val_csv, header=None, storage_options={"anon": False})
# first column = label, second = side_num (if you kept it), rest = features
y  = val.iloc[:, 0].to_numpy()
X  = val.iloc[:, 1:].to_numpy()

# optional weights (if present)
try:
    w = pd.read_csv(w_val_csv, header=None, storage_options={"anon": False}).iloc[:, 0].to_numpy()
except Exception:
    w = None

# --- Load your model (two options) ---
# Option A: if you extracted model.json locally:
# booster = xgb.Booster(model_file="model.json")

# Option B: load a saved model.bin/json you downloaded locally already:
booster = xgb.Booster()
booster.load_model("model.json")  # put model.json next to this script, or change the path

# --- Predict + metric ---
dval = xgb.DMatrix(X, label=y, weight=w)
pred = booster.predict(dval)
auprc = average_precision_score(y, pred, sample_weight=w)
print(f"AUPRC (validation): {auprc:.5f}")
print(f"pos_rate: {y.mean():.5f}, n={len(y)}")