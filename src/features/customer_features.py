"""
Customer-level feature engineering and time-aware churn target generation.
Designed with strict separation between observation and performance windows to avoid data leakage.
"""

from typing import Optional, Tuple
import numpy as np
import pandas as pd


def create_customer_features(
    df: pd.DataFrame,
    reference_date: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """
    Generate rich customer-level behavioral features from transaction history.

    Features generated:
    - Recency: Days since latest purchase
    - Frequency: Total distinct orders
    - Monetary: Total spend
    - AOV: Average Order Value
    - TotalQuantity: Total items purchased
    - BasketSize: Average items per order
    - UniqueProducts: Number of distinct stock items bought
    - ProductDiversity: Ratio of unique products to total quantity
    - TenureDays: Days since first purchase
    - LifespanDays: Days between first and last purchase
    - PurchaseVelocity: Orders per 30-day period
    - Country: Modal purchase country
    """
    if df.empty:
        raise ValueError("Transaction DataFrame is empty.")

    if "Revenue" not in df.columns:
        df = df.copy()
        df["Revenue"] = df["Quantity"] * df["UnitPrice"]

    if reference_date is None:
        ref_date = pd.to_datetime(df["InvoiceDate"]).max() + pd.Timedelta(days=1)
    else:
        ref_date = pd.to_datetime(reference_date)

    # Base customer aggregations
    grouped = df.groupby("CustomerID")

    agg_df = grouped.agg(
        FirstPurchase=("InvoiceDate", "min"),
        LastPurchase=("InvoiceDate", "max"),
        Frequency=("InvoiceNo", "nunique"),
        TotalQuantity=("Quantity", "sum"),
        Monetary=("Revenue", "sum"),
        UniqueProducts=("StockCode", "nunique"),
        Country=("Country", lambda s: s.mode().iloc[0] if not s.empty else "Unknown"),
    ).reset_index()

    # Derived metrics
    agg_df["FirstPurchase"] = pd.to_datetime(agg_df["FirstPurchase"])
    agg_df["LastPurchase"] = pd.to_datetime(agg_df["LastPurchase"])
    
    agg_df["Recency"] = (ref_date - agg_df["LastPurchase"]).dt.days
    agg_df["TenureDays"] = (ref_date - agg_df["FirstPurchase"]).dt.days
    agg_df["LifespanDays"] = (agg_df["LastPurchase"] - agg_df["FirstPurchase"]).dt.days
    
    agg_df["AOV"] = np.round(agg_df["Monetary"] / np.maximum(agg_df["Frequency"], 1), 2)
    agg_df["BasketSize"] = np.round(agg_df["TotalQuantity"] / np.maximum(agg_df["Frequency"], 1), 2)
    agg_df["ProductDiversity"] = np.round(
        agg_df["UniqueProducts"] / np.maximum(agg_df["TotalQuantity"], 1), 4
    )
    agg_df["PurchaseVelocity"] = np.round(
        (agg_df["Frequency"] / np.maximum(agg_df["TenureDays"], 1)) * 30.0, 3
    )

    # Fill any nulls
    agg_df["Monetary"] = np.round(agg_df["Monetary"], 2)
    agg_df = agg_df.fillna(0)

    return agg_df


def create_time_based_churn_dataset(
    df: pd.DataFrame,
    cutoff_date: Optional[pd.Timestamp] = None,
    observation_days: int = 90,
) -> Tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    """
    Construct a time-aware churn dataset with ZERO lookahead data leakage.

    Split logic:
    - History Window (<= cutoff_date): Features are calculated purely within this window.
    - Future Performance Window (> cutoff_date): Customer is labeled Churned (1)
      if they had 0 purchases in this window, and Retained (0) if they made >= 1 purchase.

    Parameters:
        df: Cleaned transaction history.
        cutoff_date: Split date. Defaults to max(InvoiceDate) - observation_days.
        observation_days: Length of the forward inactivity evaluation window (default 90 days).

    Returns:
        (churn_df, cutoff_date, max_date)
    """
    df = df.copy()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    max_date = df["InvoiceDate"].max()

    if cutoff_date is None:
        cutoff_date = max_date - pd.Timedelta(days=observation_days)
    else:
        cutoff_date = pd.to_datetime(cutoff_date)

    # 1. Historical transactions
    df_history = df[df["InvoiceDate"] <= cutoff_date].copy()
    if df_history.empty:
        raise ValueError(f"No transactions found on or before cutoff date: {cutoff_date}")

    # 2. Performance window transactions
    df_future = df[df["InvoiceDate"] > cutoff_date].copy()

    # Active customers during future window
    future_active_customers = set(df_future["CustomerID"].unique())

    # Build features purely from history
    features_df = create_customer_features(df_history, reference_date=cutoff_date)

    # Label target: 1 = Churned (no repurchase in future window), 0 = Active/Retained
    features_df["Churn"] = features_df["CustomerID"].apply(
        lambda cid: 0 if cid in future_active_customers else 1
    )

    return features_df, cutoff_date, max_date
