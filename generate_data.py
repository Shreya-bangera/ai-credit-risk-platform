"""Generate a synthetic dataset mimicking Home Credit Default Risk structure."""
import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)
N = 50000
Path("./data").mkdir(exist_ok=True)

# Base features
age_years = np.random.normal(43, 12, N).clip(20, 70)
income = np.random.lognormal(11.5, 0.6, N).clip(25000, 1_000_000)
employed_years = (age_years - 22) * np.random.uniform(0.3, 0.9, N)
employed_years = employed_years.clip(0, 40)

credit = income * np.random.uniform(1.5, 8, N)
credit = credit.clip(45000, 4_000_000)
annuity = credit / np.random.uniform(24, 96, N)
goods_price = credit * np.random.uniform(0.8, 1.0, N)

ext1 = np.random.beta(5, 2, N).clip(0.1, 1.0)
ext2 = np.random.beta(5, 2, N).clip(0.1, 1.0)
ext3 = np.random.beta(4, 2, N).clip(0.1, 1.0)

children = np.random.choice([0, 1, 2, 3, 4], N, p=[0.45, 0.30, 0.17, 0.06, 0.02])
fam_members = children + np.random.choice([1, 2], N, p=[0.25, 0.75])

gender = np.random.choice(["M", "F"], N, p=[0.36, 0.64])
own_car = np.random.choice(["Y", "N"], N, p=[0.44, 0.56])
own_realty = np.random.choice(["Y", "N"], N, p=[0.69, 0.31])
contract_type = np.random.choice(["Cash loans", "Revolving loans"], N, p=[0.90, 0.10])
income_type = np.random.choice(
    ["Working", "Commercial associate", "Pensioner", "State servant", "Unemployed"],
    N, p=[0.52, 0.23, 0.18, 0.06, 0.01]
)
education = np.random.choice(
    ["Secondary / secondary special", "Higher education",
     "Incomplete higher", "Lower secondary", "Academic degree"],
    N, p=[0.57, 0.31, 0.08, 0.03, 0.01]
)
family_status = np.random.choice(
    ["Married", "Single / not married", "Civil marriage", "Separated", "Widow"],
    N, p=[0.64, 0.15, 0.10, 0.07, 0.04]
)
housing_type = np.random.choice(
    ["House / apartment", "With parents", "Municipal apartment",
     "Rented apartment", "Office apartment", "Co-op apartment"],
    N, p=[0.72, 0.12, 0.08, 0.05, 0.02, 0.01]
)
region_rating = np.random.choice([1, 2, 3], N, p=[0.15, 0.68, 0.17])

# Derived
credit_income_ratio = credit / (income + 1)
annuity_income_ratio = annuity / (income + 1)
days_birth = -(age_years * 365).astype(int)
days_employed = -(employed_years * 365).astype(int)

# Add some missing values (realistic)
ext1_missing = ext1.copy().astype(float)
ext1_missing[np.random.choice(N, int(N * 0.56), replace=False)] = np.nan
ext3_missing = ext3.copy().astype(float)
ext3_missing[np.random.choice(N, int(N * 0.19), replace=False)] = np.nan

# TARGET: logistic model based on real risk drivers
log_odds = (
    -2.5
    - 3.0 * ext2
    - 2.5 * np.nan_to_num(ext3, nan=0.4)
    - 2.0 * np.nan_to_num(ext1_missing, nan=0.4)
    + 0.8 * credit_income_ratio
    + 0.6 * annuity_income_ratio
    - 0.03 * (age_years - 30)
    - 0.02 * employed_years
    + 0.4 * (region_rating == 3).astype(float)
    + 0.3 * (income_type == "Unemployed").astype(float)
    + 0.2 * (education == "Lower secondary").astype(float)
    - 0.2 * (own_realty == "Y").astype(float)
    + np.random.normal(0, 0.5, N)
)
prob_default = 1 / (1 + np.exp(-log_odds))
target = (np.random.uniform(0, 1, N) < prob_default).astype(int)

print(f"Default rate: {target.mean()*100:.1f}%")

df = pd.DataFrame({
    "SK_ID_CURR": range(100002, 100002 + N),
    "TARGET": target,
    "NAME_CONTRACT_TYPE": contract_type,
    "CODE_GENDER": gender,
    "FLAG_OWN_CAR": own_car,
    "FLAG_OWN_REALTY": own_realty,
    "CNT_CHILDREN": children,
    "AMT_INCOME_TOTAL": income.round(2),
    "AMT_CREDIT": credit.round(2),
    "AMT_ANNUITY": annuity.round(2),
    "AMT_GOODS_PRICE": goods_price.round(2),
    "NAME_INCOME_TYPE": income_type,
    "NAME_EDUCATION_TYPE": education,
    "NAME_FAMILY_STATUS": family_status,
    "NAME_HOUSING_TYPE": housing_type,
    "REGION_RATING_CLIENT": region_rating,
    "REGION_RATING_CLIENT_W_CITY": region_rating,
    "DAYS_BIRTH": days_birth,
    "DAYS_EMPLOYED": days_employed,
    "DAYS_REGISTRATION": -(np.random.uniform(0, 25 * 365, N)).astype(int),
    "DAYS_ID_PUBLISH": -(np.random.uniform(0, 15 * 365, N)).astype(int),
    "DAYS_LAST_PHONE_CHANGE": -(np.random.uniform(0, 5 * 365, N)).astype(int),
    "CNT_FAM_MEMBERS": fam_members.astype(float),
    "EXT_SOURCE_1": ext1_missing.round(6),
    "EXT_SOURCE_2": ext2.round(6),
    "EXT_SOURCE_3": ext3_missing.round(6),
    "FLAG_WORK_PHONE": np.random.choice([0, 1], N, p=[0.71, 0.29]),
    "FLAG_PHONE": np.random.choice([0, 1], N, p=[0.28, 0.72]),
    "FLAG_EMAIL": np.random.choice([0, 1], N, p=[0.91, 0.09]),
    "OBS_30_CNT_SOCIAL_CIRCLE": np.random.poisson(1.4, N).clip(0, 30).astype(float),
    "OBS_60_CNT_SOCIAL_CIRCLE": np.random.poisson(1.4, N).clip(0, 30).astype(float),
    "DEF_30_CNT_SOCIAL_CIRCLE": np.random.poisson(0.14, N).clip(0, 10).astype(float),
    "DEF_60_CNT_SOCIAL_CIRCLE": np.random.poisson(0.10, N).clip(0, 10).astype(float),
    "AMT_REQ_CREDIT_BUREAU_YEAR": np.random.poisson(1.9, N).clip(0, 25).astype(float),
})

out = "./data/application_train.csv"
df.to_csv(out, index=False)
print(f"Dataset saved to {out} — {len(df):,} rows, {len(df.columns)} columns")
