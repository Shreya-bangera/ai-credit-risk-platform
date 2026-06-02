import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("./data")

SELECTED_FEATURES = [
    "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE",
    "DAYS_BIRTH", "DAYS_EMPLOYED", "DAYS_REGISTRATION", "DAYS_ID_PUBLISH",
    "CNT_FAM_MEMBERS", "CNT_CHILDREN",
    "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
    "CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY",
    "NAME_CONTRACT_TYPE", "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE",
    "REGION_RATING_CLIENT", "REGION_RATING_CLIENT_W_CITY",
    "FLAG_WORK_PHONE", "FLAG_PHONE", "FLAG_EMAIL",
    "DEF_30_CNT_SOCIAL_CIRCLE", "DEF_60_CNT_SOCIAL_CIRCLE",
    "OBS_30_CNT_SOCIAL_CIRCLE", "OBS_60_CNT_SOCIAL_CIRCLE",
    "AMT_REQ_CREDIT_BUREAU_YEAR", "DAYS_LAST_PHONE_CHANGE",
]


def load_raw(filename="application_train.csv") -> pd.DataFrame:
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. "
            "Download from https://www.kaggle.com/competitions/home-credit-default-risk/data "
            "and place application_train.csv in the ./data folder."
        )
    return pd.read_csv(path)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / (df["AMT_INCOME_TOTAL"] + 1)
    df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / (df["AMT_INCOME_TOTAL"] + 1)
    df["CREDIT_TERM"] = df["AMT_ANNUITY"] / (df["AMT_CREDIT"] + 1)
    df["DAYS_EMPLOYED_RATIO"] = df["DAYS_EMPLOYED"] / (df["DAYS_BIRTH"] + 1)
    df["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365
    df["EMPLOYED_YEARS"] = -df["DAYS_EMPLOYED"].clip(upper=0) / 365
    return df


def preprocess(df: pd.DataFrame, fit_encoders: dict = None):
    """
    Returns (X, y, encoders).
    If fit_encoders is None, fits new label encoders; otherwise reuses them.
    """
    df = engineer_features(df)

    extra = ["CREDIT_INCOME_RATIO", "ANNUITY_INCOME_RATIO", "CREDIT_TERM",
             "DAYS_EMPLOYED_RATIO", "AGE_YEARS", "EMPLOYED_YEARS"]
    features = SELECTED_FEATURES + extra

    available = [f for f in features if f in df.columns]
    X = df[available].copy()

    cat_cols = X.select_dtypes(include="object").columns.tolist()
    encoders = fit_encoders or {}

    for col in cat_cols:
        X[col] = X[col].astype(str).fillna("Unknown")
        if col not in encoders:
            unique_vals = X[col].unique().tolist()
            encoders[col] = {v: i for i, v in enumerate(unique_vals)}
        X[col] = X[col].map(encoders[col]).fillna(-1).astype(int)

    # Fill numeric NaNs with median
    for col in X.select_dtypes(include=[np.number]).columns:
        X[col] = X[col].fillna(X[col].median())

    y = df["TARGET"] if "TARGET" in df.columns else None
    return X, y, encoders


def get_sample_for_db(n=5000) -> pd.DataFrame:
    """Return a sample of the dataset for the SQL chatbot."""
    df = load_raw()
    df = engineer_features(df)
    sample = df.sample(min(n, len(df)), random_state=42)
    return sample
