# save as ml_bot/scripts/sageMaker/find_pooled_threshold.py
import numpy as np, pandas as pd
import s3fs, sys

VAL = "s3://tradebot-config-tokyo/tmp/pred_debug/2025-11-11T01-16-00Z/predictions.csv"  # VAL
TST = "s3://tradebot-config-tokyo/tmp/pred_debug/2025-11-11T01-17-05Z/predictions.csv"  # TEST

fs = s3fs.S3FileSystem(anon=False)
def load(uri): 
    with fs.open(uri) as f: 
        df = pd.read_csv(f)
    return df.rename(columns={"p_pred":"p"}).assign(split=("VAL" if "16-00" in uri else "TEST"))

dv = load(VAL)
dt = load(TST)
df = pd.concat([dv, dt], ignore_index=True)

y = df["y"].to_numpy().astype(int)
p = df["p"].to_numpy().astype(float)

# grille = percentiles sur le pool (stable)
thr_list = np.unique(np.quantile(p, np.linspace(0,1,201)))
def f1_at(thr, y, p):
    yhat = (p >= thr).astype(int)
    tp = ((y==1)&(yhat==1)).sum()
    fp = ((y==0)&(yhat==1)).sum()
    fn = ((y==1)&(yhat==0)).sum()
    prec = tp / max(tp+fp,1)
    rec  = tp / max(tp+fn,1)
    f1 = 2*prec*rec / max(prec+rec,1e-12)
    return prec, rec, f1

best = None
for thr in thr_list:
    prec, rec, f1 = f1_at(thr, y, p)
    if not best or f1 > best["f1"]:
        best = {"thr": float(thr), "prec": float(prec), "rec": float(rec), "f1": float(f1)}

def split_metrics(thr, name, dfx):
    yy = dfx["y"].to_numpy().astype(int)
    pp = dfx["p"].to_numpy().astype(float)
    return f1_at(thr, yy, pp)

print("[pooled] best_thr=%.4f  F1=%.3f  P=%.3f  R=%.3f" % (best["thr"], best["f1"], best["prec"], best["rec"]))
for name, dfx in [("VAL", dv), ("TEST", dt)]:
    P,R,F1 = split_metrics(best["thr"], name, dfx)
    print("  - %s @thr=%.4f -> F1=%.3f  P=%.3f  R=%.3f" % (name, best["thr"], F1, P, R))