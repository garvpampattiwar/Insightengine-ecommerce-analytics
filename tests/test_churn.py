"""
Unit tests for Churn Prediction ML pipeline.
"""

import pytest
import pandas as pd
import numpy as np

from src.models.churn import train_churn_models, load_churn_artifacts, FEATURE_COLUMNS


@pytest.fixture
def sample_churn_data():
    np.random.seed(42)
    n = 120
    recency = np.random.randint(1, 180, size=n)
    # Give higher recency higher probability of churn
    churn_prob = 1 / (1 + np.exp(-(recency - 60) / 20))
    churn = (np.random.rand(n) < churn_prob).astype(int)

    # Ensure both classes exist
    churn[0] = 0
    churn[1] = 1

    return pd.DataFrame({
        "CustomerID": np.arange(1000, 1000 + n),
        "Recency": recency,
        "Frequency": np.random.randint(1, 20, size=n),
        "Monetary": np.random.uniform(50, 3000, size=n),
        "AOV": np.random.uniform(20, 200, size=n),
        "TotalQuantity": np.random.randint(5, 500, size=n),
        "BasketSize": np.random.uniform(2, 50, size=n),
        "UniqueProducts": np.random.randint(1, 30, size=n),
        "ProductDiversity": np.random.uniform(0.1, 1.0, size=n),
        "TenureDays": np.random.randint(10, 365, size=n),
        "LifespanDays": np.random.randint(0, 300, size=n),
        "PurchaseVelocity": np.random.uniform(0.1, 5.0, size=n),
        "Churn": churn,
    })


def test_train_churn_models(sample_churn_data):
    results = train_churn_models(sample_churn_data, test_size=0.25, random_state=42, save_artifacts=False)

    assert "metrics" in results
    assert "logistic_regression" in results["metrics"]
    assert "random_forest" in results["metrics"]

    rf_m = results["metrics"]["random_forest"]
    assert 0.0 <= rf_m["accuracy"] <= 1.0
    assert 0.0 <= rf_m["roc_auc"] <= 1.0
    assert 0.0 <= rf_m["f1"] <= 1.0

    assert "RiskTier" in results["scored_df"].columns
    assert "ChurnProbability" in results["scored_df"].columns
    assert set(results["scored_df"]["RiskTier"].dropna().unique()).issubset(
        {"Low Risk", "Medium Risk", "High Risk"}
    )


def test_load_churn_artifacts():
    artifacts = load_churn_artifacts()
    assert "best_model" in artifacts
    assert "logistic_regression" in artifacts
    assert "random_forest" in artifacts
    assert "scaler" in artifacts
    assert "customer_churn_scored" in artifacts
    assert len(artifacts["customer_churn_scored"]) > 0
    assert "ChurnProbability" in artifacts["customer_churn_scored"].columns
