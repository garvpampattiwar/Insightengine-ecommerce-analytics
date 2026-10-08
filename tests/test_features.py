"""
Unit tests for feature engineering (RFM, Customer features, Churn target, Product features).
"""

import pytest
import pandas as pd
import numpy as np

from src.features.rfm import calculate_rfm, preprocess_rfm_for_clustering
from src.features.customer_features import (
    create_customer_features,
    create_time_based_churn_dataset,
)
from src.features.product_features import create_product_features


@pytest.fixture
def sample_cleaned_df():
    dates = pd.date_range("2011-01-01", periods=10, freq="10D")
    return pd.DataFrame({
        "InvoiceNo": [f"INV{i}" for i in range(10)],
        "StockCode": ["P1", "P2", "P1", "P3", "P2", "P1", "P4", "P2", "P3", "P1"],
        "Description": [f"Product {c}" for c in ["P1", "P2", "P1", "P3", "P2", "P1", "P4", "P2", "P3", "P1"]],
        "Quantity": [2, 5, 1, 10, 2, 4, 1, 3, 2, 8],
        "InvoiceDate": dates,
        "UnitPrice": [10.0, 5.0, 10.0, 2.0, 5.0, 10.0, 50.0, 5.0, 2.0, 10.0],
        "CustomerID": [101, 101, 102, 103, 102, 101, 104, 102, 103, 101],
        "Country": ["United Kingdom"] * 10,
        "Revenue": [20.0, 25.0, 10.0, 20.0, 10.0, 40.0, 50.0, 15.0, 4.0, 80.0],
    })


def test_calculate_rfm(sample_cleaned_df):
    rfm = calculate_rfm(sample_cleaned_df)
    assert len(rfm) == sample_cleaned_df["CustomerID"].nunique()
    assert set(["CustomerID", "Recency", "Frequency", "Monetary", "R_Score", "F_Score", "M_Score", "RFM_Segment"]).issubset(rfm.columns)
    
    # Customer 101 has 4 invoices: 20 + 25 + 40 + 80 = 165
    c101 = rfm[rfm["CustomerID"] == 101].iloc[0]
    assert c101["Frequency"] == 4
    assert c101["Monetary"] == 165.0
    assert 1 <= c101["R_Score"] <= 5
    assert 1 <= c101["F_Score"] <= 5
    assert 1 <= c101["M_Score"] <= 5


def test_preprocess_rfm_for_clustering(sample_cleaned_df):
    rfm = calculate_rfm(sample_cleaned_df)
    scaled_data, scaler, df_log = preprocess_rfm_for_clustering(rfm)
    assert scaled_data.shape == (len(rfm), 3)
    assert np.allclose(scaled_data.mean(axis=0), 0, atol=1e-7)
    assert np.allclose(scaled_data.std(axis=0), 1, atol=1e-7)


def test_create_customer_features(sample_cleaned_df):
    cust_feat = create_customer_features(sample_cleaned_df)
    assert len(cust_feat) == sample_cleaned_df["CustomerID"].nunique()
    expected_cols = ["Recency", "Frequency", "Monetary", "AOV", "BasketSize", "UniqueProducts", "TenureDays"]
    for col in expected_cols:
        assert col in cust_feat.columns
    assert (cust_feat["AOV"] > 0).all()


def test_create_time_based_churn_dataset(sample_cleaned_df):
    # Split on 2011-03-01
    churn_df, cutoff, max_date = create_time_based_churn_dataset(
        sample_cleaned_df, cutoff_date=pd.Timestamp("2011-03-01")
    )
    assert "Churn" in churn_df.columns
    assert set(churn_df["Churn"].unique()).issubset({0, 1})
    assert cutoff < max_date


def test_create_product_features(sample_cleaned_df):
    prod_feat = create_product_features(sample_cleaned_df)
    assert len(prod_feat) == sample_cleaned_df["StockCode"].nunique()
    assert "TotalRevenue" in prod_feat.columns
    assert "RevenueRank" in prod_feat.columns
    assert prod_feat["TotalRevenue"].iloc[0] >= prod_feat["TotalRevenue"].iloc[-1]
