<div align="center">

# 🏦 NeoStats — AI-Powered Credit Risk Intelligence Platform

### Intelligence. Innovation. Impact.

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.3-02569B?style=for-the-badge&logo=microsoft&logoColor=white)](https://lightgbm.readthedocs.io)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-412991?style=for-the-badge&logo=openai&logoColor=white)](https://openai.com)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

<br/>

> An end-to-end credit risk intelligence platform built on the  
> [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk/data) dataset.  
> Covering **EDA → ML Risk Scoring → SHAP Explainability → NL→SQL Chatbot** — all in one Streamlit UI.

<br/>

![Platform Preview](https://img.shields.io/badge/Status-Production%20Ready-brightgreen?style=flat-square)
![ROC-AUC](https://img.shields.io/badge/ROC--AUC-~0.76-blue?style=flat-square)
![Dataset](https://img.shields.io/badge/Dataset-50K%20rows-orange?style=flat-square)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Project Structure](#-project-structure)
- [Features](#-features)
- [Quick Start](#-quick-start)
- [Environment Variables](#-environment-variables)
- [Module Details](#-module-details)
- [Model Performance](#-model-performance)
- [Design Decisions](#-design-decisions)
- [Known Limitations](#-known-limitations)

---

## 🎯 Overview

Banks face pressure to make **faster, more accurate, and explainable** credit decisions. NeoStats addresses real banking needs:

| Problem | Solution |
|---|---|
| Identify high-risk applicants early | ML risk scoring with probability bands |
| Explain decisions to auditors | SHAP per-prediction explainability |
| Let analysts explore data in plain English | GPT-4o-mini powered NL→SQL chatbot |
| Bridge ML insights with credit policy | Automated business rule derivation |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Streamlit UI  (app/main.py)                │
│                                                             │
│   ┌────────┐  ┌────────┐  ┌──────────┐  ┌───────┐  ┌─────┐ │
│   │  EDA   │  │  Risk  │  │  SHAP    │  │ Rules │  │ NL  │ │
│   │ Charts │  │  Pred  │  │ Explain  │  │       │  │ SQL │ │
│   └────────┘  └────────┘  └──────────┘  └───────┘  └─────┘ │
└──────────────────────────┬──────────────────────────────────┘
                           │
         ┌─────────────────┼──────────────────┐
         ▼                 ▼                  ▼
   ┌───────────┐    ┌─────────────┐    ┌────────────────┐
   │    EDA    │    │  LightGBM   │    │  OpenAI GPT    │
   │  Module   │    │  + SMOTE    │    │  NL→SQL Agent  │
   │ (Plotly)  │    │  + SHAP     │    │  + SQLite DB   │
   └───────────┘    └─────────────┘    └────────────────┘
         │                 │
         └─────────────────┘
                  │
        ┌─────────────────┐
        │ application_    │
        │  train.csv      │
        │   (./data/)     │
        └─────────────────┘
```

---

## 📁 Project Structure

```
ai-credit-risk-platform/
├── 📂 app/
│   ├── main.py                  # Streamlit UI — all 6 pages
│   └── 📂 modules/
│       ├── data_processing.py   # Feature engineering & preprocessing
│       ├── ml_model.py          # LightGBM training & inference
│       ├── explainability.py    # SHAP explanations & rule derivation
│       ├── eda.py               # EDA charts & business insights
│       └── talk_to_data.py      # NL→SQL agent (OpenAI + SQLite)
├── 📂 data/                     # Place application_train.csv here
├── 📂 models/                   # Saved model artifacts (auto-generated)
├── 📂 documents/                # Presentation PDF
├── generate_data.py             # Synthetic dataset generator
├── train.py                     # Standalone model training script
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
└── README.md
```

---

## ✨ Features

### 📊 Exploratory Data Analysis
- 7 interactive Plotly charts (distributions, scatter, boxplots, heatmaps)
- 7 actionable business insights
- Data quality summary with missing value analysis
- Dataset statistics dashboard

### 🤖 ML Risk Prediction
- Input any applicant's details via a clean form UI
- Get **default probability (0–100%)** + **risk band (Low / Medium / High)**
- Animated gauge chart with color-coded risk zones
- Recommendation: Approve / Manual Review / Decline

### 🔍 Explainable AI (SHAP)
- Per-prediction SHAP waterfall bar chart
- Feature impact table with direction (increases/decreases risk)
- Red = risk-increasing features, Green = risk-decreasing features

### 📋 Business Decision Rules
- Top-10 rules derived from model feature importances
- Human-readable format for compliance & credit policy teams
- Feature importance visualization

### 💬 Talk to Data (NL→SQL)
- Ask questions in plain English → get SQL + results
- 7 pre-built sample queries to try
- SQL safety validation (blocks DROP/DELETE/INSERT/UPDATE)
- Conversation history with clear chat

---

## 🚀 Quick Start

### Option A — Docker *(Recommended)*

```bash
# 1. Clone the repo
git clone https://github.com/Shreya-bangera/ai-credit-risk-platform.git
cd ai-credit-risk-platform

# 2. Generate or add dataset
python generate_data.py        # generates synthetic dataset
# OR place real application_train.csv in ./data/

# 3. Configure environment
cp .env.example .env
# Edit .env — set your OPENAI_API_KEY

# 4. Build and run
docker-compose up --build

# 5. Open in browser
# http://localhost:8501
```

### Option B — Local Python

```bash
# 1. Clone and setup
git clone https://github.com/Shreya-bangera/ai-credit-risk-platform.git
cd ai-credit-risk-platform

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env — set your OPENAI_API_KEY

# 4. Generate dataset
python generate_data.py

# 5. Train the model
python train.py

# 6. Launch the app
streamlit run app/main.py
```

> 💡 You can also skip `train.py` and train directly from the **Overview** page in the UI.

---

## 🔐 Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENAI_API_KEY` | ✅ Yes (chatbot) | — | OpenAI API key for NL→SQL |
| `OPENAI_MODEL` | No | `gpt-4o-mini` | LLM model to use |
| `DATA_DIR` | No | `./data` | Path to data directory |
| `MODEL_DIR` | No | `./models` | Path to model artifacts |
| `APP_PORT` | No | `8501` | Streamlit port |

---

## 📦 Module Details

### 1. Data Processing — `data_processing.py`

- Loads `application_train.csv` from `./data/`
- Engineers **6 derived features**:

| Feature | Formula |
|---|---|
| `CREDIT_INCOME_RATIO` | `AMT_CREDIT / AMT_INCOME_TOTAL` |
| `ANNUITY_INCOME_RATIO` | `AMT_ANNUITY / AMT_INCOME_TOTAL` |
| `CREDIT_TERM` | `AMT_ANNUITY / AMT_CREDIT` |
| `DAYS_EMPLOYED_RATIO` | `DAYS_EMPLOYED / DAYS_BIRTH` |
| `AGE_YEARS` | `-DAYS_BIRTH / 365` |
| `EMPLOYED_YEARS` | `-DAYS_EMPLOYED / 365` |

- Label-encodes all categoricals, median-fills numeric NaNs
- Total: **32 base + 6 engineered = 38 features**

---

### 2. Machine Learning — `ml_model.py`

**Model:** LightGBM Classifier

**Class Imbalance Strategy:**

```
Dataset: ~5-8% default rate (heavily imbalanced)

Strategy 1 — SMOTE (sampling_strategy=0.3)
  → Generates synthetic minority samples during training

Strategy 2 — scale_pos_weight=3
  → Penalizes misclassification of defaulters more in the loss function

Combined effect: Better recall on actual defaulters
```

**Key Hyperparameters:**

```python
n_estimators     = 500
learning_rate    = 0.05
num_leaves       = 63
subsample        = 0.8
colsample_bytree = 0.8
scale_pos_weight = 3
```

---

### 3. Explainable AI — `explainability.py`

- Uses `shap.TreeExplainer` — **exact** SHAP values for tree models (faster than KernelExplainer)
- Returns top-10 features ranked by absolute SHAP value
- `derive_rules()` maps feature importances → human-readable credit policy rules

---

### 4. Talk-to-Data — `talk_to_data.py`

**Pipeline:** `Natural Language → GPT-4o-mini → SQL → SQLite → DataFrame → UI`

**Prompt Engineering approach:**
- System prompt includes full table schema with column descriptions
- `temperature=0` for deterministic, reproducible SQL generation
- `max_tokens=300` to minimize API cost
- SQL safety validation blocks: `DROP, DELETE, INSERT, UPDATE, ALTER`
- Regex strips markdown backticks from model output

**Sample working queries:**
```
1. "What is the average income of applicants who defaulted vs those who didn't?"
2. "What is the default rate by education type?"
3. "How many applicants own a car and what is their default rate?"
4. "Show the top 5 income types by average credit amount"
5. "What percentage of female applicants defaulted?"
6. "What is the average age of defaulters vs non-defaulters?"
7. "Show default rate by region rating"
```

---

## 📈 Model Performance

| Metric | Value |
|---|---|
| **ROC-AUC** | ~0.76 |
| **Average Precision** | ~0.35 |
| **Brier Score** | ~0.07 |

**Risk Band Thresholds:**

| 🟢 Low Risk | 🟡 Medium Risk | 🔴 High Risk |
|---|---|---|
| Probability 0–30% | Probability 30–60% | Probability 60–100% |
| → **Approve** | → **Manual Review** | → **Decline** |

---

## 🧠 Design Decisions

| Decision | Rationale |
|---|---|
| **LightGBM over XGBoost** | Faster training, native categorical support, lower memory footprint |
| **SMOTE + scale_pos_weight** | Dual strategy addresses imbalance at both data and loss function level |
| **GPT-4o-mini for NL→SQL** | Cost-efficient (~$0.00015/query), sufficient accuracy for structured SQL; temperature=0 ensures consistency |
| **SQLite for chatbot DB** | Zero-infrastructure, file-based, perfect for 5K-row analytical queries |
| **Streamlit UI** | Fastest path to a multi-page data app — no frontend/backend split needed |
| **SHAP TreeExplainer** | Exact (not approximate) SHAP for tree models; 10-100x faster than KernelExplainer |
| **Synthetic data generator** | Allows evaluators to run the full platform without Kaggle access |

---

## ⚠️ Known Limitations & Improvements

| Limitation | Potential Improvement |
|---|---|
| Only `application_train.csv` used | Add bureau, installments, POS tables → +3-5% AUC |
| 5K-row SQLite sample for chatbot | Use full dataset with DuckDB or PostgreSQL |
| No hyperparameter tuning | Add Optuna Bayesian search → +2% AUC |
| Single-prediction SHAP only | Add global SHAP summary plots on training data |
| No authentication on UI | Add `streamlit-authenticator` for production |

---

## 📄 License

MIT © 2024

---

<div align="center">

Built with ❤️ for the NeoStats AI Engineer Assignment

</div>
