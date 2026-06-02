import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from app.modules.ml_model import load_model
from app.modules.data_processing import preprocess


def get_shap_explainer(model, X_background: pd.DataFrame):
    return shap.TreeExplainer(model, X_background)


def explain_single(input_dict: dict, top_n: int = 10):
    """
    Returns a dict with shap_values, feature_names, feature_values,
    base_value for a single prediction.
    """
    model, encoders, feature_names = load_model()
    df = pd.DataFrame([input_dict])
    X, _, _ = preprocess(df, fit_encoders=encoders)

    for col in feature_names:
        if col not in X.columns:
            X[col] = 0
    X = X[feature_names]

    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X)

    # For binary classification LightGBM, shap_values returns list [neg, pos]
    if isinstance(shap_vals, list):
        sv = shap_vals[1][0]
        base = explainer.expected_value[1]
    else:
        sv = shap_vals[0]
        base = explainer.expected_value

    indices = np.argsort(np.abs(sv))[::-1][:top_n]
    return {
        "shap_values": sv[indices].tolist(),
        "feature_names": [feature_names[i] for i in indices],
        "feature_values": X.iloc[0].values[indices].tolist(),
        "base_value": float(base),
    }


def plot_shap_bar(shap_data: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 5))
    names = shap_data["feature_names"]
    vals = shap_data["shap_values"]
    colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in vals]
    y_pos = range(len(names))
    ax.barh(y_pos, vals[::-1], color=colors[::-1])
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(names[::-1], fontsize=9)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP Value (impact on default probability)")
    ax.set_title("Feature Contributions to Risk Score")
    plt.tight_layout()
    return fig


def derive_rules(model, feature_names: list, top_n: int = 10) -> list[dict]:
    """Extract top feature importances as human-readable rules."""
    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1][:top_n]
    rules = []
    rule_templates = {
        "EXT_SOURCE_2": "Higher external credit score (EXT_SOURCE_2) → Lower default risk",
        "EXT_SOURCE_3": "Higher external credit score (EXT_SOURCE_3) → Lower default risk",
        "EXT_SOURCE_1": "Higher external credit score (EXT_SOURCE_1) → Lower default risk",
        "DAYS_BIRTH": "Older applicants (more negative DAYS_BIRTH) → Lower default risk",
        "AMT_CREDIT": "Higher loan amount → Slightly higher default risk",
        "DAYS_EMPLOYED": "Longer employment history → Lower default risk",
        "CREDIT_INCOME_RATIO": "High credit-to-income ratio → Higher default risk",
        "ANNUITY_INCOME_RATIO": "High annuity-to-income ratio → Higher default risk",
        "AMT_INCOME_TOTAL": "Higher income → Lower default risk",
        "DAYS_LAST_PHONE_CHANGE": "Recent phone change → Slightly higher risk",
    }
    for i in idx:
        fname = feature_names[i]
        rules.append({
            "feature": fname,
            "importance": round(float(importances[i]), 4),
            "rule": rule_templates.get(fname, f"{fname} is a significant predictor of default risk"),
        })
    return rules
