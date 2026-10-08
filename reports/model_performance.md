# InsightEngine: Model Performance Evaluation & Selection Report

## 1. Executive Summary
This report documents the rigorous evaluation of the predictive models developed for **InsightEngine**. In customer analytics, model evaluation cannot rely on simple classification accuracy alone. Because losing a high-value customer incurs significant commercial penalties, model selection must balance discrimination capability (**ROC-AUC**), recall (**capturing true churning accounts**), and precision (**minimizing wasted promotional expenditure on false alarms**).

Both **Logistic Regression** and **Random Forest** were trained on an audited, leak-free cohort of 3,368 customer accounts with a ground-truth forward churn rate of **43.02%** over a 90-day evaluation window.

---

## 2. Quantitative Model Benchmark Comparison
Both models underwent 5-fold cross-validation grid search (`GridSearchCV`) to optimize decision thresholds and hyperparameters. The final models were evaluated on an unseen, stratified holdout test set ($N_{\text{test}} = 842$ accounts).

| Evaluation Metric | Logistic Regression (Baseline) | Random Forest (Ensemble) | Winner / Delta |
| :--- | :--- | :--- | :--- |
| **ROC-AUC** | 0.7272 | **0.7362** | **Random Forest (+0.90%)** |
| **F1-Score** | 0.6331 | **0.6361** | **Random Forest (+0.30%)** |
| **Accuracy** | 0.6710 (67.10%) | **0.6793 (67.93%)** | **Random Forest (+0.83%)** |
| **Precision** | 0.6081 (60.81%) | **0.6211 (62.11%)** | **Random Forest (+1.30%)** |
| **Recall** | **0.6602 (66.02%)** | 0.6519 (65.19%) | Logistic Regression (+0.83%) |
| **Best Hyperparameters** | $C = 1.0$, `solver='lbfgs'` | `n_estimators=50`, `max_depth=4`, `min_samples_split=2` | Robust regularization |

---

## 3. Confusion Matrix Analysis (Holdout Test Set)

### 3.1 Random Forest Classifier (Selected Production Model)
$$\begin{array}{c|cc}
& \textbf{Predicted Retained (0)} & \textbf{Predicted Churned (1)} \\
\hline
\textbf{Actual Retained (0)} & 336\text{ (True Negative)} & 144\text{ (False Positive)} \\
\textbf{Actual Churned (1)} & 126\text{ (False Negative)} & 236\text{ (True Positive)} \\
\end{array}$$

- **True Negatives ($TN = 336$)**: Correctly identified stable, active accounts; no promotional discount needed.
- **True Positives ($TP = 236$)**: Correctly caught churners in time for proactive retention interventions.
- **False Positives ($FP = 144$)**: Accounts flagged as at-risk who would have repurchased anyway (acceptable risk: customer receives proactive marketing).
- **False Negatives ($FN = 126$)**: Churned accounts missed by the model.

### 3.2 Logistic Regression Model
$$\begin{array}{c|cc}
& \textbf{Predicted Retained (0)} & \textbf{Predicted Churned (1)} \\
\hline
\textbf{Actual Retained (0)} & 326\text{ (True Negative)} & 154\text{ (False Positive)} \\
\textbf{Actual Churned (1)} & 123\text{ (False Negative)} & 239\text{ (True Positive)} \\
\end{array}$$

---

## 4. Model Selection Rationale
**Random Forest** was selected as the primary production engine for the following verified criteria:
1. **Superior Discrimination**: Achieved a higher ROC-AUC (**0.7362** vs 0.7272), demonstrating superior capability in ranking customers across probability spectra.
2. **Reduced False Alarms**: Precision of **62.11%** (vs 60.81% for Logistic Regression) produced 10 fewer false alarms on the test set, preventing budget dilution in discount distribution.
3. **Non-linear Relationship Handling**: Random Forest naturally captures interaction effects between `PurchaseVelocity` and `Recency` without requiring manual polynomial feature engineering.

---

## 5. Feature Importance & Interpretability

### 5.1 Random Forest Feature Importance (Gini Importance)
| Rank | Feature | Importance Score | Business Interpretation |
| :---: | :--- | :---: | :--- |
| **1** | `Recency` | **0.3782** | Days since last purchase is the single strongest indicator of customer attrition. |
| **2** | `PurchaseVelocity` | **0.2315** | Decay in orders per 30-day window provides early warning prior to total dormancy. |
| **3** | `LifespanDays` | **0.1450** | Tenured customers demonstrate higher baseline inertia against churning. |
| **4** | `TenureDays` | **0.0892** | Account maturity establishes expected repurchase periodicity. |
| **5** | `Frequency` | **0.0631** | Accounts with $\ge 5$ lifetime orders show substantially reduced defection rates. |
| **6** | `BasketSize` | **0.0384** | Average units purchased per order reflects retail vs wholesale intent. |
| **7** | `UniqueProducts` | **0.0210** | Diversity of purchased catalog correlates with brand loyalty. |
| **8** | `AOV` | **0.0185** | Average order monetary value. |

---

## 6. Calibrated Churn Probability Tiers & Intervention Strategy
To operationalize predictions for non-technical commercial teams, raw probabilities are mapped to discrete risk tiers:

| Risk Tier | Probability Range | Customer Accounts | Population % | Historical Spend at Risk | Operational Action |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Low Risk** | $p < 0.35$ | 1,198 | 35.57% | $4,252,443.26 | Standard automated communications; no retention discount. |
| **Medium Risk** | $0.35 \le p \le 0.65$ | 1,462 | 43.41% | $851,697.94 | Nurture sequences with personalized product recommendations. |
| **High Risk** | $p > 0.65$ | 708 | 21.02% | $259,429.41 | **Urgent intervention**: Time-sensitive win-back coupon flow at day 45. |

---

## 7. Artifact Persistence
Serialized model artifacts are preserved in `models/churn/`:
- `best_model.pkl`: Production Random Forest classifier.
- `logistic_regression.pkl`: Trained Logistic Regression baseline.
- `random_forest.pkl`: Trained Random Forest classifier.
- `preprocessing.pkl`: Fitted `StandardScaler` transformer.
- `metadata.json`: Hyperparameters, cross-validation metrics, ROC curves, and risk counts.
- `customer_churn_scored.csv`: Complete scored customer database.
