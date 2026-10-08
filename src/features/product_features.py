"""
Product feature engineering and aggregation utilities.
"""

from typing import Optional
import numpy as np
import pandas as pd


def create_product_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate transactional data to product level.

    Metrics:
    - TotalRevenue
    - TotalUnitsSold
    - OrderCount
    - UniqueCustomers
    - AvgPrice
    - ProductShareOfRevenue
    """
    if df.empty:
        raise ValueError("DataFrame cannot be empty.")

    if "Revenue" not in df.columns:
        df = df.copy()
        df["Revenue"] = df["Quantity"] * df["UnitPrice"]

    prod_df = df.groupby(["StockCode"]).agg(
        Description=("Description", "first"),
        TotalRevenue=("Revenue", "sum"),
        TotalUnitsSold=("Quantity", "sum"),
        OrderCount=("InvoiceNo", "nunique"),
        UniqueCustomers=("CustomerID", "nunique"),
        AvgPrice=("UnitPrice", "mean"),
    ).reset_index()

    prod_df["TotalRevenue"] = np.round(prod_df["TotalRevenue"], 2)
    prod_df["AvgPrice"] = np.round(prod_df["AvgPrice"], 2)
    total_sales = prod_df["TotalRevenue"].sum()
    prod_df["RevenueSharePct"] = np.round((prod_df["TotalRevenue"] / total_sales) * 100, 3)

    # Sort by TotalRevenue descending
    prod_df = prod_df.sort_values("TotalRevenue", ascending=False).reset_index(drop=True)
    prod_df["RevenueRank"] = np.arange(1, len(prod_df) + 1)

    return prod_df
