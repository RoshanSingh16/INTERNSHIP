"""
train_model.py
Trains a Random Forest + XGBoost ensemble for fraud detection.
Run: python train_model.py
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing    import LabelEncoder, StandardScaler
from sklearn.ensemble         import RandomForestClassifier
from sklearn.metrics          import (classification_report, confusion_matrix,
                                      roc_auc_score, average_precision_score,
                                      precision_score, recall_score, f1_score)
from imblearn.over_sampling   import SMOTE
from xgboost                  import XGBClassifier

# ── paths ──────────────────────────────────────────────────────────────────────
DATA_PATH   = Path("data/creditcard.csv")
MODELS_DIR  = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

# ── load & encode ───────────────────────────────────────────────────────────────
print("📂  Loading data …")
df = pd.read_csv(DATA_PATH)

le = LabelEncoder()
df["merchant_cat"] = le.fit_transform(df["merchant_cat"])
joblib.dump(le, MODELS_DIR / "label_encoder.pkl")

FEATURES = ["amount","hour","day_of_week","merchant_cat",
            "distance_km","prev_txn_hrs","v1","v2","v3","v4","v5"]
TARGET   = "is_fraud"

X = df[FEATURES].values
y = df[TARGET].values

# ── scale ───────────────────────────────────────────────────────────────────────
scaler = StandardScaler()
X = scaler.fit_transform(X)
joblib.dump(scaler, MODELS_DIR / "scaler.pkl")

# ── split ───────────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# ── SMOTE oversampling ──────────────────────────────────────────────────────────
print("⚖️   Applying SMOTE …")
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)

# ── train Random Forest ─────────────────────────────────────────────────────────
print("🌲  Training Random Forest …")
rf = RandomForestClassifier(
    n_estimators=200, max_depth=12, min_samples_split=5,
    class_weight="balanced", random_state=42, n_jobs=-1)
rf.fit(X_res, y_res)
joblib.dump(rf, MODELS_DIR / "random_forest.pkl")

# ── train XGBoost ───────────────────────────────────────────────────────────────
print("🚀  Training XGBoost …")
scale_pos = (y_res == 0).sum() / (y_res == 1).sum()
xgb = XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    scale_pos_weight=scale_pos, use_label_encoder=False,
    eval_metric="logloss", random_state=42, n_jobs=-1)
xgb.fit(X_res, y_res)
joblib.dump(xgb, MODELS_DIR / "xgboost.pkl")

# ── evaluate ────────────────────────────────────────────────────────────────────
def evaluate(name, model, X_t, y_t):
    y_pred = model.predict(X_t)
    y_prob = model.predict_proba(X_t)[:, 1]
    print(f"\n{'─'*50}")
    print(f"  {name}")
    print(f"{'─'*50}")
    print(classification_report(y_t, y_pred, target_names=["Legit","Fraud"]))
    return {
        "model":     name,
        "roc_auc":   round(roc_auc_score(y_t, y_prob), 4),
        "avg_prec":  round(average_precision_score(y_t, y_prob), 4),
        "precision": round(precision_score(y_t, y_pred), 4),
        "recall":    round(recall_score(y_t, y_pred), 4),
        "f1":        round(f1_score(y_t, y_pred), 4),
        "conf_matrix": confusion_matrix(y_t, y_pred).tolist(),
    }

metrics = {
    "random_forest": evaluate("Random Forest", rf,  X_test, y_test),
    "xgboost":       evaluate("XGBoost",       xgb, X_test, y_test),
    "features":      FEATURES,
    "classes":       le.classes_.tolist(),
}

with open(MODELS_DIR / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("\n✅  All models & metrics saved to /models/")
print(f"    RF  ROC-AUC : {metrics['random_forest']['roc_auc']}")
print(f"    XGB ROC-AUC : {metrics['xgboost']['roc_auc']}")
