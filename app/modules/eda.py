import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from app.modules.data_processing import load_raw, engineer_features


def load_eda_data(sample_n=20000) -> pd.DataFrame:
    df = load_raw()
    df = engineer_features(df)
    return df.sample(min(sample_n, len(df)), random_state=42)


def summary_stats(df: pd.DataFrame) -> dict:
    total = len(df)
    default_rate = df["TARGET"].mean() * 100
    missing_pct = (df.isnull().sum() / total * 100).sort_values(ascending=False)
    num_features = df.select_dtypes(include=np.number).shape[1]
    cat_features = df.select_dtypes(include="object").shape[1]
    return {
        "total_records": total,
        "default_rate": round(default_rate, 2),
        "missing_pct": missing_pct,
        "num_features": num_features,
        "cat_features": cat_features,
    }


def plot_target_distribution(df: pd.DataFrame):
    counts = df["TARGET"].value_counts().reset_index()
    counts.columns = ["Target", "Count"]
    counts["Label"] = counts["Target"].map({0: "No Default", 1: "Default"})
    fig = px.pie(counts, values="Count", names="Label",
                 title="Loan Default Distribution",
                 color_discrete_sequence=["#2ecc71", "#e74c3c"])
    return fig


def plot_age_distribution(df: pd.DataFrame):
    fig = px.histogram(df, x="AGE_YEARS", color=df["TARGET"].map({0: "No Default", 1: "Default"}),
                       nbins=40, barmode="overlay", opacity=0.7,
                       title="Age Distribution by Default Status",
                       labels={"AGE_YEARS": "Age (Years)", "color": "Status"},
                       color_discrete_map={"No Default": "#2ecc71", "Default": "#e74c3c"})
    return fig


def plot_income_vs_credit(df: pd.DataFrame):
    sample = df.sample(min(3000, len(df)), random_state=1)
    fig = px.scatter(sample, x="AMT_INCOME_TOTAL", y="AMT_CREDIT",
                     color=sample["TARGET"].map({0: "No Default", 1: "Default"}),
                     opacity=0.5, title="Income vs Credit Amount",
                     labels={"AMT_INCOME_TOTAL": "Annual Income", "AMT_CREDIT": "Credit Amount"},
                     color_discrete_map={"No Default": "#2ecc71", "Default": "#e74c3c"})
    fig.update_xaxes(range=[0, 1_000_000])
    return fig


def plot_ext_source_boxplot(df: pd.DataFrame):
    melted = df[["TARGET", "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].dropna()
    melted = melted.melt(id_vars="TARGET", var_name="Source", value_name="Score")
    melted["Status"] = melted["TARGET"].map({0: "No Default", 1: "Default"})
    fig = px.box(melted, x="Source", y="Score", color="Status",
                 title="External Credit Scores by Default Status",
                 color_discrete_map={"No Default": "#2ecc71", "Default": "#e74c3c"})
    return fig


def plot_default_by_education(df: pd.DataFrame):
    grp = df.groupby("NAME_EDUCATION_TYPE")["TARGET"].mean().reset_index()
    grp.columns = ["Education", "Default Rate"]
    grp = grp.sort_values("Default Rate", ascending=True)
    fig = px.bar(grp, x="Default Rate", y="Education", orientation="h",
                 title="Default Rate by Education Level",
                 color="Default Rate", color_continuous_scale="RdYlGn_r")
    return fig


def plot_credit_income_ratio(df: pd.DataFrame):
    fig = px.histogram(df[df["CREDIT_INCOME_RATIO"] < 10],
                       x="CREDIT_INCOME_RATIO",
                       color=df[df["CREDIT_INCOME_RATIO"] < 10]["TARGET"].map(
                           {0: "No Default", 1: "Default"}),
                       nbins=50, barmode="overlay", opacity=0.7,
                       title="Credit-to-Income Ratio by Default Status",
                       color_discrete_map={"No Default": "#2ecc71", "Default": "#e74c3c"})
    return fig


def plot_missing_values(df: pd.DataFrame):
    missing = (df.isnull().sum() / len(df) * 100)
    missing = missing[missing > 0].sort_values(ascending=False).head(20)
    fig = px.bar(x=missing.values, y=missing.index, orientation="h",
                 title="Top 20 Features with Missing Values (%)",
                 labels={"x": "Missing %", "y": "Feature"})
    return fig


BUSINESS_INSIGHTS = [
    "📊 **Class Imbalance**: ~8% of applicants default, requiring SMOTE/class weighting for fair ML models.",
    "🎓 **Education Matters**: Applicants with lower secondary education default ~2x more than those with higher education.",
    "💳 **External Scores are Key**: EXT_SOURCE_1/2/3 are the strongest predictors — lower scores correlate strongly with default.",
    "👴 **Age Effect**: Younger applicants (20–30 years) show significantly higher default rates than older applicants.",
    "💰 **Credit-to-Income Risk**: Applicants with credit > 5× their annual income are at substantially higher default risk.",
    "🏠 **Housing Stability**: Applicants living with parents or in rented apartments default more than homeowners.",
    "📱 **Phone Change Signal**: Recent phone number changes correlate with higher default probability — a behavioral risk signal.",
]
