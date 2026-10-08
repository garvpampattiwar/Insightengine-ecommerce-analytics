# InsightEngine: Comprehensive Analytical Methodology & System Architecture

## 1. Project Overview & Problem Statement
Modern e-commerce platforms process millions of transactional events annually. Raw event streams, however, contain operational anomalies—unregistered guest checkouts, invoice cancellations, negative prices, damaged stock write-offs, and administrative fee codes. Without robust auditing and machine learning pipelines, business decisions suffer from distorted customer lifetime valuations and ineffective marketing spend.

**InsightEngine** provides an end-to-end data intelligence system integrating:
- Automated, audited data cleansing and validation.
- Exploratory data analysis (EDA) across univariate, bivariate, time-series, and geographic dimensions.
- Unsupervised customer segmentation using log-normalized RFM features and K-Means clustering.
- Supervised customer churn prediction with strict time-aware data partitioning to prevent lookahead leakage.
- Item-to-item collaborative filtering for personalized cross-sell product recommendations.
- Rigorous statistical hypothesis testing (Welch's t-test and Mann-Whitney U test) evaluating simulated commercial interventions.

---

## 2. Dataset Provenance & Ingestion
The system utilizes the gold-standard transactional benchmark:
- **Source**: UCI Machine Learning Repository — *Online Retail* (Identifier: 352).
- **Domain**: UK-based registered non-store online retailer operating internationally.
- **Span**: December 1, 2010 through December 9, 2011 (~374 calendar days).
- **Raw Volume**: 541,909 transactions across 8 core fields:
  - `InvoiceNo`: 6-digit integral invoice number (prefixed with 'C' for cancellations).
  - `StockCode`: 5-digit integral product identifier or alpha-numeric item code.
  - `Description`: Nominal product name.
  - `Quantity`: Integral units purchased or cancelled.
  - `InvoiceDate`: Datetime timestamp of invoice generation.
  - `UnitPrice`: Product unit price in British Pounds (£).
  - `CustomerID`: 5-digit integral customer account identifier.
  - `Country`: Nominal customer residency country.

The raw dataset is stored immutably in `data/raw/Online Retail.xlsx` and fast-cached as `data/raw/online_retail_raw.csv`. Raw data is never overwritten.

---

## 3. Data Cleaning Pipeline & Auditing Protocol
The automated cleaning module (`src.data.cleaning`) enforces strict data quality filters:

1. **Deduplication**: Audited and purged 5,268 exact duplicate rows.
2. **Cancellations**: Isolated 9,251 cancellation records (Invoice numbers prefixed with 'C') and archived them to `data/interim/cancelled_transactions.csv` for returns auditing.
3. **Value Bounds Enforcement**: Removed 1,336 non-positive quantity entries and 2,512 non-positive unit price anomalies (administrative adjustments, postage fees, damaged goods write-offs).
4. **Account Traceability**: Filtered 132,186 guest checkout rows lacking customer identification (`CustomerID == NaN`).
5. **Administrative SKU Removal**: Filtered 1,409 non-product service stock codes (`POST`, `D`, `M`, `PADS`, `DOT`, `CRUK`, `BANK CHARGES`).
6. **Country Entity Normalization**: Standardized geographic labels (e.g., `EIRE` $\rightarrow$ `Ireland`, `RSA` $\rightarrow$ `South Africa`).
7. **Metric Calculation**: Derived transactional line item revenue:
   $$\text{Revenue} = \text{Quantity} \times \text{UnitPrice}$$

### Cleaning Funnel Audit Summary
| Funnel Stage | Transactions | % of Initial | Action Taken |
| :--- | :--- | :--- | :--- |
| **Raw Ingested** | 541,909 | 100.0% | Ingestion and schema validation |
| **Deduplicated** | 536,641 | 99.03% | Purged exact duplicate transactions |
| **Cancellations Removed** | 527,390 | 97.32% | Archived negative cancellation invoices |
| **Valid Positives** | 523,542 | 96.61% | Excluded non-positive price/quantity anomalies |
| **Traceable Accounts** | 392,692 | 72.46% | Excluded untracked guest checkouts |
| **Final Audited Dataset** | **391,283** | **72.20%** | Removed admin SKU codes (`POST`, `BANK CHARGES`) |

---

## 4. Feature Engineering Methodology

### 4.1 RFM (Recency, Frequency, Monetary)
Customer behavioral vectors were synthesized on the audited transactional dataset:
- **Recency ($R$)**: Days between reference timestamp ($T_{\text{ref}} = \max(\text{InvoiceDate}) + 1\text{ day}$) and customer's latest purchase:
  $$R_u = \lfloor T_{\text{ref}} - \max_{t \in T_u} (t) \rfloor$$
- **Frequency ($F$)**: Total distinct purchase invoices:
  $$F_u = |\text{Unique Invoices}_u|$$
- **Monetary ($M$)**: Cumulative revenue generated:
  $$M_u = \sum_{i \in I_u} \text{Revenue}_i$$

Because e-commerce distributions exhibit heavy power-law right skewness, RFM dimensions were transformed via $\log(1 + x)$ and normalized using `StandardScaler` to ensure unbiased Euclidean distance calculations in geometric feature space.

### 4.2 Leak-Free Time-Aware Churn Target
To prevent lookahead data leakage:
- **Observation History Window** ($t \le 2011\text{-}09\text{-}10$): Customer behavioral attributes (Recency, Frequency, Monetary, AOV, Basket Size, Product Diversity, Lifespan, Purchase Velocity) are calculated **strictly on transactions occurring on or before the cutoff date**.
- **Forward Performance Window** ($t > 2011\text{-}09\text{-}10$): A customer is assigned binary target $y = 1$ (Churned) if they made 0 purchases in the forward 90-day window, and $y = 0$ (Retained) if they completed $\ge 1$ purchase.

---

## 5. Machine Learning Models

### 5.1 Customer Segmentation (K-Means)
- Scaled log-RFM features were evaluated across $K \in [2, 7]$ using:
  - **Inertia (Elbow Method)**: Measuring within-cluster sum of squares.
  - **Silhouette Coefficient**: Evaluating inter-cluster separation vs intra-cluster cohesion.
- $K = 4$ was selected as the defensible optimum balancing cluster granularity with executive actionability.
- Segments were named dynamically based on empirical cluster medians:
  1. **Champions** ($K=3$): High frequency, high spend, immediate recency.
  2. **Loyal Customers** ($K=2$): Consistent repeat cadence, moderate-to-high spend.
  3. **Potential Loyalists** ($K=0$): Recent buyers, lower order count, developing accounts.
  4. **At Risk / Hibernating** ($K=1$): High days of inactivity, low frequency.

### 5.2 Churn Prediction (Supervised Classification)
- Evaluated 3,368 active historical accounts ($43.02\%$ baseline churn rate).
- Stratified 75/25 Train/Test split preserving class distribution.
- Models Trained:
  1. **Logistic Regression**: Linear baseline with `StandardScaler` pipeline; 5-fold cross-validated hyperparameter grid over regularization parameter $C \in [0.01, 0.1, 1.0, 10.0]$.
  2. **Random Forest Classifier**: Ensemble decision tree model; 5-fold cross-validated over `n_estimators`, `max_depth`, and `min_samples_split`.
- Evaluated via ROC-AUC, F1-score, Recall, Precision, Accuracy, and Confusion Matrices.

### 5.3 Collaborative Filtering Recommendation Engine
- **Item-to-Item Cosine Similarity**: Constructs user-item interaction matrix $X \in \mathbb{R}^{U \times I}$ using log-dampened quantities:
  $$r_{u, i} = \log(1 + \text{Quantity}_{u, i})$$
- Pairwise cosine similarity matrix $S \in \mathbb{R}^{I \times I}$:
  $$S_{i, j} = \frac{\mathbf{x}_{:, i} \cdot \mathbf{x}_{:, j}}{\|\mathbf{x}_{:, i}\|_2 \|\mathbf{x}_{:, j}\|_2}$$
- **Scoring**: For customer $u$ with purchased items $I_u$, score for candidate item $k \notin I_u$ is:
  $$\hat{r}_{u, k} = \frac{\sum_{j \in I_u} S_{k, j} \times r_{u, j}}{\sum_{j \in I_u} S_{k, j} + \epsilon}$$
- **Cold-Start Fallback**: Unknown customer IDs or accounts with zero historical interactions fall back to global popularity bestsellers ranked by verified order frequency.

---

## 6. Statistical Hypothesis Testing
To validate simulated commercial feature rollouts:
- **Experiment Scenario**: Comparing control basket sizes against personalized recommendation treatments.
- **Assumptions Checked**:
  - Sample size sufficiency ($n_A, n_B \ge 1,500$).
  - Homogeneity of variance assessed via **Levene's test**.
  - Skewness resilience assessed via parametric **Welch's t-test** (unequal variances) and non-parametric **Mann-Whitney U test**.
  - Standardized effect size measured via **Cohen's d**.
  - 95% Confidence Interval computed for difference in means.

---

## 7. Limitations & Reproducibility Guarantees
- **Geographic Representation**: 82.85% of transactions originate within the United Kingdom; models may require recalibration before deploying to international subsidiaries.
- **Product Domain**: High proportion of gift and boutique wholesale buyers creates occasional bimodal basket sizes.
- **Pipeline Reproducibility**: The complete analytical pipeline is reproducible by executing `python -m src.pipeline` and testable via `python -m pytest tests`.
