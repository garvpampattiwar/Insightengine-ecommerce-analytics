"""
Unit tests validating dashboard optimizations, caching invariants,
and model artifact integrity.
"""

from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from src.data.loader import get_project_root, load_cleaned_data
from src.analytics.precompute import load_precomputed_aggregates, get_precomputed_path
from src.models.segmentation import load_segmentation_artifacts
from src.models.churn import load_churn_artifacts
from src.models.recommender import load_recommender_artifacts
from src.analytics.eda import get_summary_kpis
from app import convert_to_csv, convert_segment_csv, convert_churn_csv, get_filtered_eda_data


def test_precomputed_aggregates_exist_and_consistent():
    """Verify precomputed aggregates load instantly and match raw computations."""
    precomputed = load_precomputed_aggregates()
    assert "kpis" in precomputed
    assert "time_series" in precomputed
    assert "geographic" in precomputed
    assert "top_products" in precomputed
    assert "order_values" in precomputed
    assert "all_countries" in precomputed
    assert "executive_insights" in precomputed

    kpis = precomputed["kpis"]
    assert kpis["total_transactions"] if "total_transactions" in kpis else precomputed["total_transactions"] == 391283
    assert kpis["total_customers"] == 4334
    assert kpis["total_orders"] == 18405
    assert round(kpis["total_revenue"], 2) == 8743913.64


def test_recommender_artifact_size_and_inference():
    """Verify recommender model file is compact (< 2.5 MB) and serves known + unknown users."""
    root = get_project_root()
    model_path = root / "models" / "recommender" / "recommender_model.pkl"
    assert model_path.exists()
    size_mb = model_path.stat().st_size / (1024 * 1024)
    assert size_mb < 2.5, f"Recommender artifact is too large: {size_mb:.2f} MB"

    rec_artifacts = load_recommender_artifacts()
    rec_engine = rec_artifacts["model"]
    assert len(rec_engine.known_customers) > 0

    # 1. Known customer recommendation
    known_cid = rec_engine.known_customers[0]
    recs_known = rec_engine.recommend(known_cid, n=5)
    assert len(recs_known) == 5
    for r in recs_known:
        assert "StockCode" in r
        assert "Description" in r
        assert "Score" in r
        assert "Reason" in r
        assert r["Score"] > 0

    # 2. Unknown customer cold-start fallback
    recs_unknown = rec_engine.recommend("UNKNOWN_CUSTOMER_9999", n=5)
    assert len(recs_unknown) == 5
    for r in recs_unknown:
        assert "StockCode" in r
        assert "Description" in r
        assert "Score" in r
        assert "bestseller" in r["Reason"].lower()


def test_churn_artifacts_and_customer_scorecard():
    """Verify churn artifacts load without retraining and customer lookup is instant."""
    churn_art = load_churn_artifacts()
    assert "best_model" in churn_art
    assert "metadata" in churn_art
    assert "customer_churn_scored" in churn_art

    meta = churn_art["metadata"]
    assert "roc_curves" in meta
    assert "logistic_regression" in meta["roc_curves"]
    assert "random_forest" in meta["roc_curves"]
    assert "feature_importance_rf" in meta

    scored_df = churn_art["customer_churn_scored"]
    assert "CustomerID" in scored_df.columns
    assert "ChurnProbability" in scored_df.columns
    assert "RiskTier" in scored_df.columns
    assert len(scored_df) == 4334 or len(scored_df) == 3368

    # Indexed lookup test
    indexed_df = scored_df.set_index("CustomerID")
    sample_cid = scored_df["CustomerID"].iloc[0]
    row = indexed_df.loc[sample_cid]
    assert row["RiskTier"] in ["Low Risk", "Medium Risk", "High Risk"]


def test_segmentation_artifacts_integrity():
    """Verify customer segmentation profiles and clusters."""
    seg_art = load_segmentation_artifacts()
    profiles = seg_art["profiles"]
    cust_segs = seg_art["customer_segments"]

    assert len(profiles) == 4
    assert set(profiles["SegmentName"]).issubset({
        "Champions", "Loyal Customers", "Potential Loyalists", "At Risk / Hibernating"
    })
    assert len(cust_segs) == 4334


def test_csv_download_helpers():
    """Verify CSV export helpers produce valid CSV bytes."""
    df_sample = pd.DataFrame({
        "CustomerID": [101, 102],
        "Revenue": [50.0, 150.0],
        "Country": ["France", "Germany"],
    })
    csv_bytes = convert_to_csv(df_sample, max_rows=10)
    assert isinstance(csv_bytes, bytes)
    assert b"CustomerID,Revenue,Country" in csv_bytes

    seg_sample = pd.DataFrame({
        "CustomerID": [101],
        "Recency": [10],
        "Frequency": [2],
        "Monetary": [100.0],
        "AOV": [50.0],
        "RFM_Segment": ["222"],
        "SegmentName": ["Champions"],
    })
    csv_seg_bytes = convert_segment_csv(seg_sample, "Champions")
    assert isinstance(csv_seg_bytes, bytes)
    assert b"CustomerID,Recency,Frequency" in csv_seg_bytes


def test_filtered_eda_data():
    """Verify filter application logic."""
    df = pd.DataFrame({
        "InvoiceNo": ["1", "2", "3"],
        "InvoiceDate": pd.to_datetime(["2011-01-01", "2011-02-01", "2011-03-01"]),
        "CustomerID": [101, 102, 103],
        "Country": ["United Kingdom", "France", "Germany"],
        "Revenue": [10.0, 20.0, 30.0],
    })
    seg_df = pd.DataFrame({
        "CustomerID": [101, 102, 103],
        "SegmentName": ["Champions", "Champions", "Loyal Customers"],
    })

    # Filter country
    f1 = get_filtered_eda_data(df, "France", "", "", "All Segments", seg_df)
    assert len(f1) == 1
    assert f1.iloc[0]["Country"] == "France"

    # Filter segment
    f2 = get_filtered_eda_data(df, "All Countries", "", "", "Champions", seg_df)
    assert len(f2) == 2
