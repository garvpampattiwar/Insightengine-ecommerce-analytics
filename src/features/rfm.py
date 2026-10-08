"""
RFM (Recency, Frequency, Monetary) calculation and feature engineering.
"""

from typing import Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def calculate_rfm(
    df: pd.DataFrame,
    reference_date: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """
    Calculate customer-level RFM metrics.

    Parameters:
        df: Cleaned transaction DataFrame.
        reference_date: Reference point for calculating recency (defaults to df['InvoiceDate'].max() + 1 day).

    Returns:
        DataFrame indexed by CustomerID with Recency, Frequency, Monetary,
        R_Score, F_Score, M_Score, RFM_Segment, and RFM_Score.
    """
    if df.empty:
        raise ValueError("Cannot calculate RFM on an empty DataFrame.")

    if "Revenue" not in df.columns:
        df = df.copy()
        df["Revenue"] = df["Quantity"] * df["UnitPrice"]

    if reference_date is None:
        reference_date = pd.to_datetime(df["InvoiceDate"]).max() + pd.Timedelta(days=1)
    else:
        reference_date = pd.to_datetime(reference_date)

    # Group by customer
    rfm = df.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda d: (reference_date - pd.to_datetime(d).max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("Revenue", "sum"),
        FirstPurchase=("InvoiceDate", "min"),
        LatestPurchase=("InvoiceDate", "max"),
        TotalItems=("Quantity", "sum"),
        UniqueProducts=("StockCode", "nunique"),
    ).reset_index()

    # Round Monetary and compute AOV
    rfm["Monetary"] = np.round(rfm["Monetary"], 2)
    rfm["AOV"] = np.round(rfm["Monetary"] / rfm["Frequency"], 2)

    # Compute quantile scores (1 to 5)
    # For Recency: lower days is better -> reverse bins (5 is best/most recent)
    rfm["R_Score"] = pd.qcut(
        rfm["Recency"].rank(method="first"), q=5, labels=[5, 4, 3, 2, 1]
    ).astype(int)

    # For Frequency: higher is better
    rfm["F_Score"] = pd.qcut(
        rfm["Frequency"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5]
    ).astype(int)

    # For Monetary: higher is better
    rfm["M_Score"] = pd.qcut(
        rfm["Monetary"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5]
    ).astype(int)

    rfm["RFM_Segment"] = (
        rfm["R_Score"].astype(str)
        + rfm["F_Score"].astype(str)
        + rfm["M_Score"].astype(str)
    )
    rfm["RFM_Score"] = rfm["R_Score"] + rfm["F_Score"] + rfm["M_Score"]

    return rfm


def preprocess_rfm_for_clustering(
    rfm_df: pd.DataFrame,
    scaler: Optional[StandardScaler] = None,
) -> Tuple[np.ndarray, StandardScaler, pd.DataFrame]:
    """
    Log-transform and scale RFM values for clustering algorithms.
    Handles right-skewness and scale disparity.

    Returns:
        (scaled_array, fitted_scaler, transformed_df)
    """
    features = ["Recency", "Frequency", "Monetary"]
    for col in features:
        if col not in rfm_df.columns:
            raise ValueError(f"Missing required RFM column: {col}")

    # Log1p transformation to handle skew and ensure non-negative inputs
    # Add offset if negative values exist
    rfm_transformed = pd.DataFrame(index=rfm_df.index)
    for col in features:
        min_val = rfm_df[col].min()
        if min_val <= 0:
            rfm_transformed[f"{col}_log"] = np.log1p(rfm_df[col] - min_val + 1)
        else:
            rfm_transformed[f"{col}_log"] = np.log1p(rfm_df[col])

    log_features = [f"{col}_log" for col in features]
    
    if scaler is None:
        scaler = StandardScaler()
        scaled_array = scaler.fit_transform(rfm_transformed[log_features])
    else:
        scaled_array = scaler.transform(rfm_transformed[log_features])

    return scaled_array, scaler, rfm_transformed
