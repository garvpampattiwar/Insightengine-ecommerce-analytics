# InsightEngine — E-Commerce Customer Analytics Dashboard

[![Python Version](https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-emerald.svg)](LICENSE)
[![Tests Status](https://img.shields.io/badge/pytest-22%20passed-brightgreen.svg)](tests/)

**InsightEngine** is an enterprise-grade, end-to-end data science and machine learning system that transforms raw e-commerce transaction logs into predictive customer lifetime value, automated RFM segmentation, leak-free churn intelligence, collaborative filtering recommendations, and executive-ready strategic insights.

---

## 1. Problem Statement
In modern transactional retail, operational databases accumulate millions of records filled with noise: guest sessions lacking account IDs, returned/cancelled items, anomalous unit prices, damaged goods write-offs, and administrative fee codes.

Treating all accounts uniformly results in diluted marketing budgets, silent customer attrition, and missed basket expansion opportunities. Businesses require an automated intelligence engine capable of:
1. Auditing and cleaning transactional noise.
2. Segmenting customers according to verified buying behaviors.
3. Detecting impending churn before accounts lapse into permanent dormancy.
4. Delivering personalized product recommendations to grow Average Order Value (AOV).
5. Measuring commercial interventions through rigorous statistical hypothesis testing.

---

## 2. Project Objectives
- **Data Engineering**: Process and audit 540,000+ raw transactions into a reproducible, clean analytical dataset.
- **Exploratory Data Analysis**: Uncover seasonal cycles, geographic dependencies, order velocities, and catalog concentration.
- **Customer Segmentation**: Implement K-Means clustering on normalized RFM features, evaluated via Elbow and Silhouette methods.
- **Predictive Churn Intelligence**: Train Random Forest and Logistic Regression models using strict forward-window time partitioning to eliminate lookahead data leakage.
- **Recommendation Engine**: Develop an Item-to-Item Collaborative Filtering system backed by cold-start bestseller fallbacks.
- **Statistical Rigor**: Conduct A/B testing simulations with Welch's t-test and Mann-Whitney U tests.
- **Enterprise Dashboard**: Deliver a responsive, 8-page interactive Streamlit analytics platform.

---

## 3. Dataset Provenance
- **Dataset**: UCI Machine Learning Repository — *Online Retail* (Identifier: 352).
- **Domain**: UK-based registered non-store online retailer operating internationally.
- **Timeline**: December 1, 2010 – December 9, 2011 (~374 days).
- **Raw Volume**: 541,909 transactions across 8 attributes (`InvoiceNo`, `StockCode`, `Description`, `Quantity`, `InvoiceDate`, `UnitPrice`, `CustomerID`, `Country`).
- **Storage**: Unaltered raw files reside in `data/raw/` (`Online Retail.xlsx` and fast-cached `online_retail_raw.csv`).

---

## 4. End-to-End System Architecture

```text
Dataset (UCI Online Retail)
   │
   ▼
Data Loading & Schema Validation (src.data.loader)
   │
   ▼
Data Cleaning & Audit (src.data.cleaning)
   ├── Filter Cancellations ('C' Invoices)
   ├── Filter Non-positive Quantity & Price
   ├── Remove Missing CustomerID
   └── Calculate Revenue = Quantity × UnitPrice
   │
   ▼
Processed Dataset (data/processed/cleaned_ecommerce.csv)
   │
   ├──────────────────────────────┬──────────────────────────────┬──────────────────────────────┐
   ▼                              ▼                              ▼                              ▼
Exploratory Data             RFM Feature                    Time-Aware Churn               Collaborative
Analysis (src.analytics)     Engineering                    Engineering                    Filtering Recommender
   │                              │                              │                              │
   ▼                              ▼                              ▼                              ▼
Interactive Trends           K-Means Clustering             Random Forest & LogReg         Item-Item Cosine
& Distributions              (Optimal K, Profiles)          (GridSearchCV, ROC-AUC)        Similarity & Fallback
   │                              │                              │                              │
   └──────────────────────────────┴──────────────────────────────┴──────────────────────────────┘
                                                  │
                                                  ▼
                                       Statistical Testing (A/B)
                                                  │
                                                  ▼
                                     Streamlit Dashboard (app.py)
                                                  │
                                                  ▼
                                    Actionable Business Insights
```

---

## 5. Repository Structure

```text
InsightEngine/
├── app.py                      # Main 8-page Streamlit dashboard
├── README.md                   # Complete system documentation
├── requirements.txt            # Pinned production dependencies
├── pytest.ini                  # Pytest configuration
├── .gitignore                  # Production git ignore rules
├── LICENSE                     # MIT Open Source License
├── Procfile                    # Cloud deployment process file
├── runtime.txt                 # Python runtime version
├── .env.example                # Sample environment variables
│
├── .streamlit/
│   └── config.toml             # Custom enterprise UI theme
│
├── data/
│   ├── raw/                    # Immutable raw dataset
│   ├── interim/                # Cancelled invoices archive
│   ├── processed/              # Cleaned e-commerce dataset (CSV, Parquet, Precomputed joblib)
│   └── external/               # Reference datasets
│
├── notebooks/
│   ├── 01_data_cleaning.ipynb  # Executed: Data inspection & cleaning
│   ├── 02_eda.ipynb            # Executed: Univariate, bivariate & time series
│   ├── 03_segmentation.ipynb   # Executed: RFM & K-Means clustering
│   ├── 04_churn.ipynb          # Executed: Time-aware ML churn modeling
│   └── 05_recommender.ipynb    # Executed: Collaborative filtering
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py           # Ingestion, schema validation & parquet fast-path
│   │   └── cleaning.py         # Multi-step cleaning pipeline & audit
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   ├── rfm.py              # RFM calculation & log-scaling
│   │   ├── customer_features.py# Time-aware features & churn target
│   │   └── product_features.py # Catalog aggregation metrics
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── segmentation.py     # K-Means training & profiling
│   │   ├── churn.py            # Supervised modeling & risk scoring
│   │   └── recommender.py      # Item-item collaborative filtering
│   │
│   ├── analytics/
│   │   ├── __init__.py
│   │   ├── eda.py              # KPI & distribution aggregations
│   │   ├── statistics.py       # A/B test simulation & hypothesis tests
│   │   ├── business_insights.py# Automated executive insights engine
│   │   └── precompute.py       # High-performance aggregate caching (<80 KB)
│   │
│   └── visualization/
│   │   ├── __init__.py
│   │   └── charts.py           # Accessible Plotly interactive chart builders
│   │
├── models/
│   ├── segmentation/           # Serialized kmeans.pkl, scaler.pkl, metadata.json
│   ├── churn/                  # Serialized best_model.pkl, lr.pkl, rf.pkl, metadata.json
│   └── recommender/            # Serialized recommender_model.pkl, catalogs (<2 MB)
│
├── reports/
│   ├── methodology.md          # Technical methodology and architecture
│   ├── model_performance.md    # Metrics benchmark and model selection
│   ├── business_insights.md    # Data-driven executive findings
│   ├── presentation_outline.md # 7-minute executive presentation with speaker notes
│   └── pipeline_run_summary.json# Verified run metrics
│
├── scripts/
│   ├── benchmark_loaders.py    # Ingestion benchmark suite
│   ├── profile_dashboard.py    # Runtime performance profiling
│   ├── generate_notebooks.py   # Notebook builder script
│   └── execute_notebooks.py    # Notebook execution script
│
└── tests/
    ├── test_data_pipeline.py   # Data ingestion & cleaning tests
    ├── test_features.py        # RFM & churn target tests
    ├── test_segmentation.py    # K-Means clustering tests
    ├── test_churn.py           # ML classification tests
    ├── test_recommender.py     # Collaborative filtering tests
    └── test_dashboard_optimization.py # Caching, latency & UI integrity tests
```

---

## 6. Installation & Quickstart

### Prerequisites
- Python 3.11 or Python 3.12
- Git

### 1. Clone & Set Up Environment
```bash
git clone https://github.com/your-username/InsightEngine.git
cd InsightEngine
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run Complete Training Pipeline
To verify and train all models from scratch:
```bash
python -m src.pipeline
```

### 3. Run Test Suite
```bash
python -m pytest tests
```
*Output: 22 passed tests in <35s.*

### 4. Launch Streamlit Analytics Dashboard
```bash
streamlit run app.py
```
The application will open locally at `http://localhost:8501`.

---

## 7. Empirical Analytical Findings

### 7.1 Data Cleaning Funnel
- **Initial Volume**: 541,909 raw records.
- **Audited Volume**: 391,283 verified transactions (**72.20% retention**).
- **Verified Metrics**: **$8,743,913.64** net revenue across **4,334 customers** and **18,405 orders** (Average Order Value: **$475.08**).

### 7.2 Customer Segments (K-Means on RFM)
| Segment | Customers | Pop. % | Total Revenue | Rev. % | Avg Recency | Avg Frequency | Avg Spend |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Champions** | 684 | **15.78%** | **$5,598,068.68** | **64.02%** | 11.7 days | 14.0 orders | $8,184.31 |
| **Loyal Customers** | 1,174 | **27.09%** | **$2,123,880.77** | **24.29%** | 67.0 days | 4.2 orders | $1,809.10 |
| **Potential Loyalists**| 840 | **19.38%** | **$446,683.58** | **5.11%** | 18.5 days | 2.1 orders | $531.77 |
| **At Risk / Hibernating** | 1,636 | **37.75%** | **$575,280.61** | **6.58%** | 183.1 days | 1.3 orders | $351.64 |

### 7.3 Churn Prediction Models (90-Day Forward Inactivity)
- **Evaluated Cohort**: 3,368 accounts with a baseline churn rate of **43.02%**.
- **Model Comparison (Holdout Test Set, $N = 842$)**:
  - **Random Forest (Selected)**: **ROC-AUC: 0.7362**, **F1: 0.6361**, **Accuracy: 67.93%**, **Precision: 62.11%**, **Recall: 65.19%**.
  - **Logistic Regression**: **ROC-AUC: 0.7272**, **F1: 0.6331**, **Accuracy: 67.10%**, **Precision: 60.81%**, **Recall: 66.02%**.
- **Identified Risk Tiers**:
  - **High Risk ($p > 0.65$)**: 708 accounts (21.02%) representing **$259,429.41** in revenue exposed to defection.
  - **Medium Risk ($0.35 \le p \le 0.65$)**: 1,462 accounts (43.41%).
  - **Low Risk ($p < 0.35$)**: 1,198 accounts (35.57%).

### 7.4 Recommendation Engine Metrics & Compact Architecture
- **Architecture**: Item-to-Item Collaborative Filtering using sparse cosine similarity nearest-neighbor graphs.
- **Model Footprint**: **1.95 MB** (`models/recommender/recommender_model.pkl`), down from 182.63 MB (**98.9% size reduction**), eliminating Git LFS requirements and enabling instant GitHub pushes.
- **Catalog Coverage**: **30.26%** across evaluated user cohorts with 346 distinct items dynamically recommended.
- **Inference Latency**: Sub-millisecond lookup per customer query.

### 7.5 A/B Experiment Simulation
- **Control Basket Mean**: $402.73 | **Treatment Basket Mean**: $432.73
- **Observed Lift**: +$30.00 (+7.45%)
- **Welch's t-test**: $p = 0.0504$ | **Mann-Whitney U**: $p = 0.0120$ (Statistically significant rank-sum shift) | **Cohen's d**: $0.0715$.

### 7.6 High-Performance Optimization Architecture
- **Precomputed Analytical Cache**: Precomputed metrics and aggregations (`dashboard_precomputed_aggregates.joblib`, 77 KB) eliminate repetitive groupbys across 391k rows, providing instant sub-second page switches.
- **Parquet Ingestion Fast-Path**: Dual CSV/Parquet storage accelerates cold data loading by ~4x (220 ms vs 750+ ms).
- **Streamlit UI Fragments (`@st.fragment`)**: Interactive sliders (e.g., A/B simulation lift, recommendation counts) re-render locally without triggering full dashboard re-runs.
- **Accessible High-Contrast Visualizations**: Chart layouts configured with high contrast labels, zero clipping margins, left-aligned titles, and Plotly Dark theme.

---

## 8. Deployment Strategy (Streamlit Cloud)
InsightEngine is designed for zero-configuration deployment to **Streamlit Community Cloud**:
1. Fork or push this repository to GitHub.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Select the repository, set Branch to `main`, and Main file path to `app.py`.
4. Streamlit Cloud automatically builds from `requirements.txt` and starts `app.py`.
5. Pre-serialized models in `models/` load in `< 2 seconds`.

---

## 9. License
Distributed under the MIT License. See [LICENSE](LICENSE) for more information.
