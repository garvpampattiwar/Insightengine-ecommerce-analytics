"""
Precomputation module for fast dashboard startup and instant page rendering.
Precomputes and caches customer-level summaries, KPI values, time-series,
geographic aggregations, top products, and executive insights.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import joblib

from ..data.loader import get_project_root, load_cleaned_data
from ..analytics.eda import (
    get_summary_kpis,
    get_time_series_aggregations,
    get_geographic_aggregations,
    get_top_products_summary,
)
from ..analytics.business_insights import generate_executive_insights


PRECOMPUTED_FILE_NAME = "dashboard_precomputed_aggregates.joblib"


def get_precomputed_path(root: Optional[Path] = None) -> Path:
    if root is None:
        root = get_project_root()
    return root / "data" / "processed" / PRECOMPUTED_FILE_NAME


def precompute_dashboard_aggregates(
    df: Optional[pd.DataFrame] = None,
    save: bool = True,
) -> Dict[str, Any]:
    """
    Precompute all dashboard aggregates once to avoid repeated expensive groupbys
    across 391,283 transactions during runtime navigation.
    """
    root = get_project_root()
    if df is None:
        df = load_cleaned_data()

    # Ensure required helper columns exist
    if "Date" not in df.columns:
        df["Date"] = df["InvoiceDate"].dt.date
    if "YearMonth" not in df.columns:
        df["YearMonth"] = df["InvoiceDate"].dt.strftime("%Y-%m")
    if "DayOfWeek" not in df.columns:
        df["DayOfWeek"] = df["InvoiceDate"].dt.day_name()
    if "Hour" not in df.columns:
        df["Hour"] = df["InvoiceDate"].dt.hour

    kpis = get_summary_kpis(df)
    ts = get_time_series_aggregations(df)
    geo = get_geographic_aggregations(df)
    top_products = get_top_products_summary(df, top_n=20)
    order_values = df.groupby("InvoiceNo")["Revenue"].sum().values

    # Invariant metadata for filters
    all_countries = sorted(df["Country"].unique().tolist())
    min_date = df["InvoiceDate"].min().date()
    max_date = df["InvoiceDate"].max().date()

    # Precompute executive insights if segmentation and churn models are present
    seg_file = root / "models" / "segmentation" / "customer_segments.csv"
    churn_file = root / "models" / "churn" / "customer_churn_scored.csv"
    
    seg_df = pd.read_csv(seg_file) if seg_file.exists() else None
    churn_df = pd.read_csv(churn_file) if churn_file.exists() else None
    
    exec_insights = generate_executive_insights(df, segmented_df=seg_df, churn_df=churn_df)

    aggregates = {
        "kpis": kpis,
        "time_series": ts,
        "geographic": geo,
        "top_products": top_products,
        "order_values": order_values,
        "all_countries": all_countries,
        "min_date": min_date,
        "max_date": max_date,
        "total_transactions": len(df),
        "total_revenue": kpis["total_revenue"],
        "total_customers": kpis["total_customers"],
        "executive_insights": exec_insights,
    }

    if save:
        out_path = get_precomputed_path(root)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(aggregates, out_path, compress=3)

    return aggregates


def load_precomputed_aggregates(force_recompute: bool = False) -> Dict[str, Any]:
    """
    Load precomputed aggregates from disk.
    If not yet computed, automatically runs precompute_dashboard_aggregates.
    """
    root = get_project_root()
    path = get_precomputed_path(root)

    if not force_recompute and path.exists():
        try:
            return joblib.load(path)
        except Exception:
            pass

    return precompute_dashboard_aggregates(save=True)
