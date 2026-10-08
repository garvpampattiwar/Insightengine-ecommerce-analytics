"""
Exploratory Data Analysis and Statistical Summaries module.
Provides Univariate, Bivariate, Time-Series, and Geographic aggregations.
"""

from typing import Dict, Any, List
import numpy as np
import pandas as pd


def get_summary_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute high-level executive KPIs."""
    if "Revenue" not in df.columns:
        df = df.copy()
        df["Revenue"] = df["Quantity"] * df["UnitPrice"]

    total_revenue = float(round(df["Revenue"].sum(), 2))
    total_customers = int(df["CustomerID"].nunique())
    total_orders = int(df["InvoiceNo"].nunique())
    total_products = int(df["StockCode"].nunique())
    avg_order_value = float(round(total_revenue / max(total_orders, 1), 2))
    avg_units_per_order = float(round(df["Quantity"].sum() / max(total_orders, 1), 1))

    return {
        "total_revenue": total_revenue,
        "total_customers": total_customers,
        "total_orders": total_orders,
        "total_products": total_products,
        "avg_order_value": avg_order_value,
        "avg_units_per_order": avg_units_per_order,
        "date_min": str(df["InvoiceDate"].min()),
        "date_max": str(df["InvoiceDate"].max()),
    }


def get_time_series_aggregations(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Compute daily, monthly, day-of-week, and hourly aggregations.
    """
    df = df.copy()
    if not np.issubdtype(df["InvoiceDate"].dtype, np.datetime64):
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    df["Date"] = df["InvoiceDate"].dt.date
    df["YearMonth"] = df["InvoiceDate"].dt.to_period("M").astype(str)
    df["DayOfWeek"] = df["InvoiceDate"].dt.day_name()
    df["Hour"] = df["InvoiceDate"].dt.hour

    daily = df.groupby("Date").agg(
        Revenue=("Revenue", "sum"),
        Orders=("InvoiceNo", "nunique"),
        Customers=("CustomerID", "nunique"),
        ItemsSold=("Quantity", "sum"),
    ).reset_index()
    daily["Date"] = pd.to_datetime(daily["Date"])

    monthly = df.groupby("YearMonth").agg(
        Revenue=("Revenue", "sum"),
        Orders=("InvoiceNo", "nunique"),
        Customers=("CustomerID", "nunique"),
        ItemsSold=("Quantity", "sum"),
    ).reset_index()

    dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dow = df.groupby("DayOfWeek").agg(
        Revenue=("Revenue", "sum"),
        Orders=("InvoiceNo", "nunique"),
    ).reindex(dow_order).dropna().reset_index()

    hourly = df.groupby("Hour").agg(
        Revenue=("Revenue", "sum"),
        Orders=("InvoiceNo", "nunique"),
    ).reset_index()

    return {
        "daily": daily,
        "monthly": monthly,
        "day_of_week": dow,
        "hourly": hourly,
    }


def get_geographic_aggregations(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate revenue, order count, and customer count by country."""
    geo = df.groupby("Country").agg(
        Revenue=("Revenue", "sum"),
        Orders=("InvoiceNo", "nunique"),
        Customers=("CustomerID", "nunique"),
        UnitsSold=("Quantity", "sum"),
    ).reset_index()

    geo["Revenue"] = np.round(geo["Revenue"], 2)
    geo["AOV"] = np.round(geo["Revenue"] / np.maximum(geo["Orders"], 1), 2)
    total_rev = geo["Revenue"].sum()
    geo["RevenuePct"] = np.round((geo["Revenue"] / total_rev) * 100, 2)

    return geo.sort_values("Revenue", ascending=False).reset_index(drop=True)


def get_top_products_summary(df: pd.DataFrame, top_n: int = 10) -> Dict[str, pd.DataFrame]:
    """Retrieve top products by Revenue, Units Sold, and Order Count."""
    prod = df.groupby(["StockCode", "Description"]).agg(
        TotalRevenue=("Revenue", "sum"),
        TotalQuantity=("Quantity", "sum"),
        OrderCount=("InvoiceNo", "nunique"),
    ).reset_index()

    prod["TotalRevenue"] = np.round(prod["TotalRevenue"], 2)

    by_rev = prod.sort_values("TotalRevenue", ascending=False).head(top_n).reset_index(drop=True)
    by_qty = prod.sort_values("TotalQuantity", ascending=False).head(top_n).reset_index(drop=True)
    by_orders = prod.sort_values("OrderCount", ascending=False).head(top_n).reset_index(drop=True)

    return {
        "by_revenue": by_rev,
        "by_quantity": by_qty,
        "by_orders": by_orders,
    }


def get_bivariate_correlations(features_df: pd.DataFrame) -> pd.DataFrame:
    """Compute Pearson correlation matrix of numerical features."""
    num_df = features_df.select_dtypes(include=[np.number])
    return num_df.corr().round(3)
