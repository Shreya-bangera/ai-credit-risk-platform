import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(
    page_title="NeoStats Credit Risk Platform",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar Navigation ──────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/fluency/96/bank-building.png", width=60)
st.sidebar.title("NeoStats Credit Risk")
st.sidebar.caption("Intelligence. Innovation. Impact.")

PAGES = ["🏠 Overview", "📊 EDA", "🤖 Risk Prediction", "🔍 Explainability", "📋 Decision Rules", "💬 Talk to Data"]
page = st.sidebar.radio("Navigate", PAGES)

# ── Helper: check data ───────────────────────────────────────────────────────
def data_available():
    from pathlib import Path
    return (Path("./data/application_train.csv")).exists()

def model_available():
    from pathlib import Path
    return (Path("./models/lgbm_model.pkl")).exists()


# ════════════════════════════════════════════════════════════════════════════
# PAGE 1: OVERVIEW
# ════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.title("🏦 AI-Powered Credit Risk Intelligence Platform")
    st.markdown("""
    > **NeoStats** | Built for the Home Credit Default Risk Challenge

    This platform provides end-to-end credit risk intelligence:
    """)

    col1, col2, col3 = st.columns(3)
    col1.metric("ML Model", "LightGBM + SMOTE", "ROC-AUC ~0.76")
    col2.metric("Explainability", "SHAP Values", "Per-prediction")
    col3.metric("Talk-to-Data", "NL → SQL", "GPT-4o-mini")

    st.markdown("---")
    st.subheader("Platform Modules")
    modules = {
        "📊 EDA": "Explore dataset statistics, distributions, and 7 business insights",
        "🤖 Risk Prediction": "Input applicant data → get risk score + band (Low/Medium/High)",
        "🔍 Explainability": "SHAP waterfall chart showing which features drove the prediction",
        "📋 Decision Rules": "Business-readable rules derived from model feature importances",
        "💬 Talk to Data": "Ask natural language questions → get SQL-backed answers",
    }
    for name, desc in modules.items():
        st.markdown(f"**{name}** — {desc}")

    st.markdown("---")
    if not data_available():
        st.warning("⚠️ Dataset not found. Place `application_train.csv` in the `./data/` folder.")
    else:
        st.success("✅ Dataset loaded")

    if not model_available():
        st.warning("⚠️ Model not trained yet.")
        if st.button("🚀 Train Model Now") and data_available():
            with st.spinner("Training LightGBM model (this takes ~2-3 minutes)..."):
                from app.modules.ml_model import train
                metrics = train()
            st.success(f"✅ Model trained! ROC-AUC: {metrics['roc_auc']}")
            st.text(metrics["classification_report"])
    else:
        st.success("✅ Model ready")


# ════════════════════════════════════════════════════════════════════════════
# PAGE 2: EDA
# ════════════════════════════════════════════════════════════════════════════
elif page == "📊 EDA":
    st.title("📊 Exploratory Data Analysis")

    if not data_available():
        st.error("Dataset not found. Place `application_train.csv` in `./data/`.")
        st.stop()

    with st.spinner("Loading data..."):
        from app.modules.eda import (
            load_eda_data, summary_stats, plot_target_distribution,
            plot_age_distribution, plot_income_vs_credit, plot_ext_source_boxplot,
            plot_default_by_education, plot_credit_income_ratio,
            plot_missing_values, BUSINESS_INSIGHTS
        )
        df = load_eda_data()
        stats = summary_stats(df)

    # Summary metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records", f"{stats['total_records']:,}")
    c2.metric("Default Rate", f"{stats['default_rate']}%")
    c3.metric("Numeric Features", stats['num_features'])
    c4.metric("Categorical Features", stats['cat_features'])

    st.markdown("---")
    st.subheader("💡 Key Business Insights")
    for insight in BUSINESS_INSIGHTS:
        st.markdown(insight)

    st.markdown("---")
    st.subheader("Visualizations")

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Target", "Age", "Income vs Credit", "Ext Scores",
        "Education", "Credit Ratio", "Missing Values"
    ])
    with tab1:
        st.plotly_chart(plot_target_distribution(df), use_container_width=True)
    with tab2:
        st.plotly_chart(plot_age_distribution(df), use_container_width=True)
    with tab3:
        st.plotly_chart(plot_income_vs_credit(df), use_container_width=True)
    with tab4:
        st.plotly_chart(plot_ext_source_boxplot(df), use_container_width=True)
    with tab5:
        st.plotly_chart(plot_default_by_education(df), use_container_width=True)
    with tab6:
        st.plotly_chart(plot_credit_income_ratio(df), use_container_width=True)
    with tab7:
        st.plotly_chart(plot_missing_values(df), use_container_width=True)

    st.markdown("---")
    st.subheader("Data Quality Summary")
    missing = stats["missing_pct"][stats["missing_pct"] > 0].head(15)
    st.dataframe(missing.reset_index().rename(columns={"index": "Feature", 0: "Missing %"}))


# ════════════════════════════════════════════════════════════════════════════
# PAGE 3: RISK PREDICTION
# ════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Risk Prediction":
    st.title("🤖 Credit Risk Prediction")

    if not model_available():
        st.error("Model not trained. Go to Overview and train the model first.")
        st.stop()

    st.markdown("Enter applicant details to get a risk score and band.")

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Personal Info")
            gender = st.selectbox("Gender", ["M", "F"])
            age = st.slider("Age (years)", 18, 70, 35)
            education = st.selectbox("Education", [
                "Higher education", "Secondary / secondary special",
                "Incomplete higher", "Lower secondary", "Academic degree"
            ])
            family_status = st.selectbox("Family Status", [
                "Married", "Single / not married", "Civil marriage",
                "Separated", "Widow"
            ])
            children = st.number_input("Number of Children", 0, 10, 0)
            own_car = st.selectbox("Owns Car", ["Y", "N"])
            own_realty = st.selectbox("Owns Realty", ["Y", "N"])

        with col2:
            st.subheader("Financial Info")
            income = st.number_input("Annual Income ($)", 10000, 1000000, 150000, step=5000)
            credit = st.number_input("Loan Amount ($)", 10000, 4000000, 500000, step=10000)
            annuity = st.number_input("Monthly Annuity ($)", 1000, 100000, 25000, step=500)
            goods_price = st.number_input("Goods Price ($)", 0, 4000000, 450000, step=10000)
            income_type = st.selectbox("Income Type", [
                "Working", "Commercial associate", "Pensioner",
                "State servant", "Unemployed"
            ])
            contract_type = st.selectbox("Contract Type", ["Cash loans", "Revolving loans"])

        with col3:
            st.subheader("Credit History")
            ext1 = st.slider("External Score 1", 0.0, 1.0, 0.5, 0.01)
            ext2 = st.slider("External Score 2", 0.0, 1.0, 0.5, 0.01)
            ext3 = st.slider("External Score 3", 0.0, 1.0, 0.5, 0.01)
            employed_years = st.slider("Years Employed", 0, 40, 5)
            region_rating = st.selectbox("Region Rating", [1, 2, 3])
            housing_type = st.selectbox("Housing Type", [
                "House / apartment", "With parents", "Municipal apartment",
                "Rented apartment", "Office apartment", "Co-op apartment"
            ])

        submitted = st.form_submit_button("🔍 Assess Risk", use_container_width=True)

    if submitted:
        input_data = {
            "CODE_GENDER": gender,
            "DAYS_BIRTH": -age * 365,
            "NAME_EDUCATION_TYPE": education,
            "NAME_FAMILY_STATUS": family_status,
            "CNT_CHILDREN": children,
            "CNT_FAM_MEMBERS": children + 2,
            "FLAG_OWN_CAR": own_car,
            "FLAG_OWN_REALTY": own_realty,
            "AMT_INCOME_TOTAL": income,
            "AMT_CREDIT": credit,
            "AMT_ANNUITY": annuity,
            "AMT_GOODS_PRICE": goods_price,
            "NAME_INCOME_TYPE": income_type,
            "NAME_CONTRACT_TYPE": contract_type,
            "EXT_SOURCE_1": ext1,
            "EXT_SOURCE_2": ext2,
            "EXT_SOURCE_3": ext3,
            "DAYS_EMPLOYED": -employed_years * 365,
            "REGION_RATING_CLIENT": region_rating,
            "REGION_RATING_CLIENT_W_CITY": region_rating,
            "NAME_HOUSING_TYPE": housing_type,
            "DAYS_REGISTRATION": -5 * 365,
            "DAYS_ID_PUBLISH": -3 * 365,
            "FLAG_WORK_PHONE": 0,
            "FLAG_PHONE": 1,
            "FLAG_EMAIL": 0,
            "DEF_30_CNT_SOCIAL_CIRCLE": 0,
            "DEF_60_CNT_SOCIAL_CIRCLE": 0,
            "OBS_30_CNT_SOCIAL_CIRCLE": 0,
            "OBS_60_CNT_SOCIAL_CIRCLE": 0,
            "AMT_REQ_CREDIT_BUREAU_YEAR": 1,
            "DAYS_LAST_PHONE_CHANGE": -365,
        }

        from app.modules.ml_model import predict_single
        result = predict_single(input_data)
        prob = result["probability"]
        band = result["risk_band"]

        st.markdown("---")
        st.subheader("📊 Risk Assessment Result")

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Default Probability", f"{prob * 100:.1f}%")
        col_b.metric("Risk Band", band)
        col_c.metric("Recommendation",
                     "✅ Approve" if band == "Low" else ("⚠️ Review" if band == "Medium" else "❌ Decline"))

        # Gauge chart
        color = "#2ecc71" if band == "Low" else ("#f39c12" if band == "Medium" else "#e74c3c")
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            title={"text": "Default Risk Score (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 30], "color": "#d5f5e3"},
                    {"range": [30, 60], "color": "#fef9e7"},
                    {"range": [60, 100], "color": "#fadbd8"},
                ],
                "threshold": {"line": {"color": "red", "width": 4}, "thickness": 0.75, "value": 60},
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

        # Store for explainability page
        st.session_state["last_input"] = input_data
        st.session_state["last_result"] = result
        st.info("💡 Go to **Explainability** tab to see which features drove this prediction.")


# ════════════════════════════════════════════════════════════════════════════
# PAGE 4: EXPLAINABILITY
# ════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Explainability":
    st.title("🔍 Explainable AI — SHAP Analysis")

    if not model_available():
        st.error("Model not trained. Go to Overview and train the model first.")
        st.stop()

    if "last_input" not in st.session_state:
        st.info("Run a prediction first on the **Risk Prediction** page.")
        st.stop()

    input_data = st.session_state["last_input"]
    result = st.session_state["last_result"]

    st.markdown(f"**Last Prediction:** Probability = `{result['probability']*100:.1f}%` | Band = `{result['risk_band']}`")

    with st.spinner("Computing SHAP values..."):
        from app.modules.explainability import explain_single, plot_shap_bar
        shap_data = explain_single(input_data)

    st.subheader("Feature Contributions (SHAP)")
    fig = plot_shap_bar(shap_data)
    st.pyplot(fig)

    st.markdown("---")
    st.subheader("Feature Impact Table")
    impact_df = pd.DataFrame({
        "Feature": shap_data["feature_names"],
        "Value": [round(v, 4) for v in shap_data["feature_values"]],
        "SHAP Impact": [round(v, 4) for v in shap_data["shap_values"]],
        "Direction": ["↑ Increases Risk" if v > 0 else "↓ Decreases Risk" for v in shap_data["shap_values"]],
    })
    st.dataframe(impact_df, use_container_width=True)

    st.markdown(f"""
    **How to read this:**
    - 🔴 Red bars = features that **increase** default probability
    - 🟢 Green bars = features that **decrease** default probability
    - Base value (average model output): `{shap_data['base_value']:.4f}`
    """)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 5: DECISION RULES
# ════════════════════════════════════════════════════════════════════════════
elif page == "📋 Decision Rules":
    st.title("📋 Business Decision Rules")

    if not model_available():
        st.error("Model not trained. Go to Overview and train the model first.")
        st.stop()

    from app.modules.ml_model import load_model
    from app.modules.explainability import derive_rules

    model, _, feature_names = load_model()
    rules = derive_rules(model, feature_names)

    st.markdown("These rules are derived from the model's feature importances and domain knowledge.")
    st.markdown("---")

    for i, rule in enumerate(rules, 1):
        with st.expander(f"Rule {i}: {rule['feature']} (importance: {rule['importance']})"):
            st.markdown(f"**{rule['rule']}**")
            st.progress(min(rule['importance'] / rules[0]['importance'], 1.0))

    st.markdown("---")
    st.subheader("Risk Band Thresholds")
    threshold_df = pd.DataFrame({
        "Risk Band": ["Low", "Medium", "High"],
        "Probability Range": ["0% – 30%", "30% – 60%", "60% – 100%"],
        "Recommendation": ["Approve", "Manual Review", "Decline"],
        "Color": ["🟢", "🟡", "🔴"],
    })
    st.dataframe(threshold_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Feature Importance Chart")
    import plotly.express as px
    rules_df = pd.DataFrame(rules)
    fig = px.bar(rules_df.sort_values("importance"), x="importance", y="feature",
                 orientation="h", title="Top Feature Importances",
                 color="importance", color_continuous_scale="Blues")
    st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 6: TALK TO DATA
# ════════════════════════════════════════════════════════════════════════════
elif page == "💬 Talk to Data":
    st.title("💬 Talk to Data — Natural Language SQL")

    if not data_available():
        st.error("Dataset not found. Place `application_train.csv` in `./data/`.")
        st.stop()

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key or api_key == "your_openai_api_key_here":
        st.warning("⚠️ Set your `OPENAI_API_KEY` in the `.env` file to use this feature.")
        st.code("OPENAI_API_KEY=sk-...")
        st.stop()

    # Initialize DB
    with st.spinner("Initializing database..."):
        from app.modules.talk_to_data import init_db, answer_question
        init_db()

    st.markdown("Ask questions about the credit dataset in plain English.")

    # Sample questions
    st.subheader("💡 Try these questions:")
    sample_qs = [
        "What is the average income of applicants who defaulted vs those who didn't?",
        "How many applicants own a car and what is their default rate?",
        "What is the default rate by education type?",
        "Show the top 5 income types by average credit amount",
        "What percentage of female applicants defaulted?",
        "What is the average age of defaulters vs non-defaulters?",
        "Show default rate by region rating",
    ]

    cols = st.columns(2)
    for i, q in enumerate(sample_qs):
        if cols[i % 2].button(q, key=f"q{i}"):
            st.session_state["chat_input"] = q

    st.markdown("---")

    # Chat history
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    user_input = st.chat_input("Ask a question about the data...")
    if "chat_input" in st.session_state:
        user_input = st.session_state.pop("chat_input")

    if user_input:
        with st.spinner("Generating SQL and querying..."):
            result = answer_question(user_input)
        st.session_state["chat_history"].append({"q": user_input, "r": result})

    # Display history
    for item in reversed(st.session_state["chat_history"]):
        with st.chat_message("user"):
            st.write(item["q"])
        with st.chat_message("assistant"):
            if item["r"]["error"]:
                st.error(f"Error: {item['r']['error']}")
            else:
                st.code(item["r"]["sql"], language="sql")
                if item["r"]["data"] is not None and not item["r"]["data"].empty:
                    st.dataframe(item["r"]["data"], use_container_width=True)
                else:
                    st.info("Query returned no results.")

    if st.button("🗑️ Clear History"):
        st.session_state["chat_history"] = []
        st.rerun()
