# train_evaluate.py
from pathlib import Path
import json, numpy as np
from config import MODEL_ARTIFACTS_DIR, log_header
from models import train_all_targets, load_trained_targets, predict_proba as _predict_proba
from progress_utils import Timer, heartbeat
from blending import learn_blend_weights, BLEND_WEIGHTS_JSON

def evaluate_models(models_dir: Path = MODEL_ARTIFACTS_DIR) -> dict:
    from models import _load_features
    df=_load_features(); metrics={}; models=load_trained_targets(models_dir)
    for t,m in models.items():
        if t not in df.columns: continue
        sub=df.dropna(subset=[t]); 
        if sub.empty: continue
        preds=_predict_proba({t:m},sub)[t]; y=sub[t].astype("category").cat.codes.values
        from sklearn.metrics import log_loss
        ll=log_loss(y,preds,labels=np.arange(preds.shape[1]))
        oh=np.eye(preds.shape[1])[y]; br=float(np.mean((oh-preds)**2))
        metrics[t]={"log_loss":float(ll),"brier":br,"n":int(len(sub))}
    return metrics

def run_training_and_eval():
    with Timer("Training all targets"): 
        train_all_targets(MODEL_ARTIFACTS_DIR)
    with Timer("Evaluating models"): 
        metrics=evaluate_models(MODEL_ARTIFACTS_DIR)
    out=MODEL_ARTIFACTS_DIR/"metrics.json"
    with open(out,"w") as f: json.dump(metrics,f,indent=2)
    print(f"Wrote metrics -> {out}")
    with Timer("Learning blend weights (ML vs DC)"):
        weights = learn_blend_weights()
    print(f"Wrote blend weights -> {BLEND_WEIGHTS_JSON}")
    heartbeat("Training, evaluation, and blending complete.")

if __name__=="__main__": 
    run_training_and_eval()
