import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix,
    average_precision_score, brier_score_loss
)
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE

from app.modules.data_processing import load_raw, preprocess

MODEL_DIR = Path("./models")
MODEL_PATH = MODEL_DIR / "lgbm_model.pkl"
ENCODER_PATH = MODEL_DIR / "encoders.pkl"
FEATURE_PATH = MODEL_DIR / "feature_names.pkl"


def train():
    MODEL_DIR.mkdir(exist_ok=True)
    print("Loading data...")
    df = load_raw()

    X, y, encoders = preprocess(df)
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Applying SMOTE for class imbalance...")
    sm = SMOTE(random_state=42, sampling_strategy=0.3)
    X_res, y_res = sm.fit_resample(X_train, y_train)

    print("Training LightGBM...")
    model = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=3,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(X_res, y_res, eval_set=[(X_test, y_test)])

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
        "avg_precision": round(average_precision_score(y_test, y_prob), 4),
        "brier_score": round(brier_score_loss(y_test, y_prob), 4),
        "classification_report": classification_report(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    joblib.dump(model, MODEL_PATH)
    joblib.dump(encoders, ENCODER_PATH)
    joblib.dump(feature_names, FEATURE_PATH)
    print(f"Model saved. ROC-AUC: {metrics['roc_auc']}")
    return metrics


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model not found. Run training first.")
    model = joblib.load(MODEL_PATH)
    encoders = joblib.load(ENCODER_PATH)
    feature_names = joblib.load(FEATURE_PATH)
    return model, encoders, feature_names


def predict_single(input_dict: dict):
    model, encoders, feature_names = load_model()
    df = pd.DataFrame([input_dict])
    X, _, _ = preprocess(df, fit_encoders=encoders)

    # Align columns
    for col in feature_names:
        if col not in X.columns:
            X[col] = 0
    X = X[feature_names]

    prob = model.predict_proba(X)[0][1]
    band = "Low" if prob < 0.3 else ("Medium" if prob < 0.6 else "High")
    return {"probability": round(float(prob), 4), "risk_band": band}


def get_risk_band(prob: float) -> str:
    if prob < 0.3:
        return "Low"
    elif prob < 0.6:
        return "Medium"
    return "High"
