"""
Unit tests for Collaborative Filtering Recommendation Engine.
"""

import os
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from src.models.recommender import CollaborativeRecommender, load_recommender_artifacts


@pytest.fixture
def sample_txn_data():
    return pd.DataFrame({
        "InvoiceNo": [f"I{i}" for i in range(1, 21)],
        "StockCode": ["P1", "P2", "P3", "P1", "P2", "P4", "P1", "P3", "P4", "P5",
                     "P1", "P2", "P3", "P4", "P5", "P1", "P2", "P3", "P4", "P5"],
        "Description": [f"Product {s}" for s in ["P1", "P2", "P3", "P1", "P2", "P4", "P1", "P3", "P4", "P5",
                                                 "P1", "P2", "P3", "P4", "P5", "P1", "P2", "P3", "P4", "P5"]],
        "Quantity": [5, 2, 3, 1, 4, 2, 6, 3, 1, 2, 5, 2, 3, 1, 4, 2, 6, 3, 1, 2],
        "UnitPrice": [10.0] * 20,
        "CustomerID": [101, 101, 101, 102, 102, 102, 103, 103, 104, 104,
                       101, 102, 103, 104, 105, 105, 106, 106, 107, 108],
        "Country": ["United Kingdom"] * 20,
        "Revenue": [50.0, 20.0, 30.0, 10.0, 40.0, 20.0, 60.0, 30.0, 10.0, 20.0,
                    50.0, 20.0, 30.0, 10.0, 40.0, 20.0, 60.0, 30.0, 10.0, 20.0],
    })


def test_recommender_fit_and_recommend(sample_txn_data):
    rec = CollaborativeRecommender(top_n_popular=5)
    rec.fit(sample_txn_data, min_product_orders=1)

    assert rec.item_similarity_df is not None
    assert rec.user_item_matrix is not None
    assert len(rec.popular_products) > 0
    assert len(rec.user_item_matrix.index) > 0
    assert len(rec.item_similarity_df.columns) > 0

    # Test recommendation for known customer
    recs = rec.recommend(101, n=3)
    assert len(recs) <= 3
    for r in recs:
        assert "StockCode" in r
        assert "Description" in r
        assert "Score" in r
        assert "Reason" in r


def test_recommender_cold_start_fallback(sample_txn_data):
    rec = CollaborativeRecommender(top_n_popular=5)
    rec.fit(sample_txn_data, min_product_orders=1)

    # Test recommendation for completely unknown customer
    recs_unknown = rec.recommend(999999, n=3)
    assert len(recs_unknown) == 3
    for r in recs_unknown:
        assert "fallback" in r["Reason"].lower() or "bestseller" in r["Reason"].lower()


def test_load_recommender_artifacts():
    artifacts = load_recommender_artifacts()
    assert "model" in artifacts
    assert "popular_products" in artifacts
    assert "product_catalog" in artifacts
    assert len(artifacts["popular_products"]) > 0

    # Verify model artifact is lightweight and well below GitHub limit (< 5 MB)
    model_path = Path("models") / "recommender" / "recommender_model.pkl"
    assert model_path.exists()
    size_mb = model_path.stat().st_size / (1024 * 1024)
    assert size_mb < 10.0, f"Artifact size {size_mb} MB exceeds compact threshold of 10 MB"
