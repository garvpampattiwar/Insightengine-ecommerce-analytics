"""
Unit tests for customer segmentation modeling and profiling.
"""

import pytest
import pandas as pd
import numpy as np

from src.models.segmentation import (
    evaluate_kmeans_clusters,
    train_segmentation_model,
    load_segmentation_artifacts,
)


@pytest.fixture
def sample_rfm():
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "CustomerID": np.arange(1000, 1000 + n),
        "Recency": np.random.randint(1, 365, size=n),
        "Frequency": np.random.randint(1, 30, size=n),
        "Monetary": np.random.uniform(50, 5000, size=n).round(2),
        "R_Score": np.random.randint(1, 6, size=n),
        "F_Score": np.random.randint(1, 6, size=n),
        "M_Score": np.random.randint(1, 6, size=n),
        "RFM_Score": np.random.randint(3, 16, size=n),
    })


def test_evaluate_kmeans_clusters(sample_rfm):
    from src.features.rfm import preprocess_rfm_for_clustering
    scaled, _, _ = preprocess_rfm_for_clustering(sample_rfm)
    eval_res = evaluate_kmeans_clusters(scaled, k_range=range(2, 5))
    assert "k_values" in eval_res
    assert len(eval_res["inertias"]) == 3
    assert len(eval_res["silhouette_scores"]) == 3
    assert 2 <= eval_res["recommended_k_silhouette"] <= 4


def test_train_segmentation_model(sample_rfm):
    model, scaler, clustered, profiles, eval_res = train_segmentation_model(
        sample_rfm, n_clusters=4, random_state=42, save_artifacts=False
    )
    assert len(clustered) == len(sample_rfm)
    assert "Cluster" in clustered.columns
    assert "SegmentName" in clustered.columns
    assert len(profiles) == 4
    assert set(["SegmentName", "CustomerCount", "CustomerPct", "RevenuePct", "RecommendedAction"]).issubset(profiles.columns)
    assert np.isclose(profiles["CustomerPct"].sum(), 100.0, atol=0.5)


def test_load_segmentation_artifacts():
    artifacts = load_segmentation_artifacts()
    assert "model" in artifacts
    assert "scaler" in artifacts
    assert "profiles" in artifacts
    assert "customer_segments" in artifacts
    assert len(artifacts["profiles"]) >= 3
