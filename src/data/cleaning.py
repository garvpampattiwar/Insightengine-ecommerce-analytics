"""
Data cleaning pipeline for InsightEngine E-Commerce dataset.
Transforms raw transactions into a clean, audited, and enriched dataset.
"""

import json
from pathlib import Path
from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd

from .loader import get_project_root, load_raw_data


def clean_online_retail(
    df_raw: pd.DataFrame,
    save_interim: bool = True,
    save_processed: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Perform end-to-end cleaning on the raw Online Retail dataset.

    Steps:
    1. Assess initial missing values and shape.
    2. Remove duplicate rows.
    3. Separate cancellation transactions (InvoiceNo starting with 'C').
    4. Remove invalid transactions (Quantity <= 0, UnitPrice <= 0).
    5. Handle missing CustomerID (drop rows without CustomerID for customer analytics).
    6. Normalize and clean text columns (Description, StockCode, Country).
    7. Standardize data types (CustomerID as int, dates parsed).
    8. Calculate Revenue = Quantity * UnitPrice.
    9. Engineer calendar and time attributes.
    10. Compile before-and-after audit statistics.

    Returns:
        (df_cleaned, audit_dict)
    """
    root = get_project_root()
    df = df_raw.copy()
    initial_shape = df.shape
    initial_missing = df.isnull().sum().to_dict()

    # Step 1: Duplicate analysis
    duplicate_count = int(df.duplicated().sum())
    df = df.drop_duplicates()

    # Step 2: Separate cancellations (starts with 'C')
    df["InvoiceNo_str"] = df["InvoiceNo"].astype(str).str.strip()
    is_cancellation = df["InvoiceNo_str"].str.upper().str.startswith("C")
    cancellations_count = int(is_cancellation.sum())

    if save_interim:
        df_cancellations = df[is_cancellation].copy()
        interim_dir = root / "data" / "interim"
        interim_dir.mkdir(parents=True, exist_ok=True)
        df_cancellations.to_csv(interim_dir / "cancelled_transactions.csv", index=False)

    df_valid_invoices = df[~is_cancellation].copy()

    # Step 3: Handle Quantity and UnitPrice validity
    invalid_quantity_count = int((df_valid_invoices["Quantity"] <= 0).sum())
    invalid_price_count = int((df_valid_invoices["UnitPrice"] <= 0).sum())

    df_valid_values = df_valid_invoices[
        (df_valid_invoices["Quantity"] > 0) & (df_valid_invoices["UnitPrice"] > 0)
    ].copy()

    # Step 4: Handle Customer ID
    missing_customer_count = int(df_valid_values["CustomerID"].isnull().sum())
    df_with_customer = df_valid_values.dropna(subset=["CustomerID"]).copy()

    # Convert CustomerID to integer
    df_with_customer["CustomerID"] = df_with_customer["CustomerID"].astype(int)

    # Step 5: Clean and normalize text columns
    # Descriptions
    df_with_customer["Description"] = (
        df_with_customer["Description"].astype(str).str.strip().str.upper()
    )
    # StockCode
    df_with_customer["StockCode"] = (
        df_with_customer["StockCode"].astype(str).str.strip().str.upper()
    )

    # Filter out non-product administrative codes if present
    admin_stock_codes = ["POST", "D", "M", "PADS", "DOT", "CRUK", "BANK CHARGES"]
    non_product_count = int(df_with_customer["StockCode"].isin(admin_stock_codes).sum())
    df_clean = df_with_customer[~df_with_customer["StockCode"].isin(admin_stock_codes)].copy()

    # Normalize Country
    country_mapping = {
        "EIRE": "Ireland",
        "RSA": "South Africa",
        "European Community": "European Union",
        "USA": "United States",
    }
    df_clean["Country"] = (
        df_clean["Country"].astype(str).str.strip().replace(country_mapping)
    )

    # Step 6: Create Revenue and Temporal Attributes
    df_clean["Revenue"] = np.round(df_clean["Quantity"] * df_clean["UnitPrice"], 2)
    df_clean["InvoiceDate"] = pd.to_datetime(df_clean["InvoiceDate"])
    df_clean["InvoiceDateOnly"] = df_clean["InvoiceDate"].dt.date
    df_clean["YearMonth"] = df_clean["InvoiceDate"].dt.to_period("M").astype(str)
    df_clean["Year"] = df_clean["InvoiceDate"].dt.year
    df_clean["Month"] = df_clean["InvoiceDate"].dt.month
    df_clean["DayOfWeek"] = df_clean["InvoiceDate"].dt.day_name()
    df_clean["Hour"] = df_clean["InvoiceDate"].dt.hour

    # Clean up auxiliary column
    df_clean = df_clean.drop(columns=["InvoiceNo_str"])

    # Final summary statistics
    final_shape = df_clean.shape
    unique_customers = int(df_clean["CustomerID"].nunique())
    unique_products = int(df_clean["StockCode"].nunique())
    unique_orders = int(df_clean["InvoiceNo"].nunique())
    total_revenue = float(np.round(df_clean["Revenue"].sum(), 2))
    date_min = str(df_clean["InvoiceDate"].min())
    date_max = str(df_clean["InvoiceDate"].max())

    audit = {
        "initial_rows": initial_shape[0],
        "initial_columns": initial_shape[1],
        "initial_missing": initial_missing,
        "duplicates_removed": duplicate_count,
        "cancellations_separated": cancellations_count,
        "invalid_quantity_removed": invalid_quantity_count,
        "invalid_price_removed": invalid_price_count,
        "missing_customer_id_removed": missing_customer_count,
        "admin_codes_removed": non_product_count,
        "final_rows": final_shape[0],
        "final_columns": final_shape[1],
        "retention_rate_pct": round((final_shape[0] / initial_shape[0]) * 100, 2),
        "unique_customers": unique_customers,
        "unique_products": unique_products,
        "unique_orders": unique_orders,
        "total_revenue": total_revenue,
        "date_range": {"min": date_min, "max": date_max},
    }

    if save_processed:
        processed_dir = root / "data" / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)
        csv_path = processed_dir / "cleaned_ecommerce.csv"
        df_clean.to_csv(csv_path, index=False)
        
        # Save audit report as JSON
        with open(processed_dir / "cleaning_audit.json", "w") as f:
            json.dump(audit, f, indent=2)

    return df_clean, audit


if __name__ == "__main__":
    print("Executing data cleaning pipeline...")
    raw_df = load_raw_data()
    cleaned_df, audit_summary = clean_online_retail(raw_df)
    print("\n=== CLEANING AUDIT SUMMARY ===")
    for k, v in audit_summary.items():
        print(f"  {k}: {v}")
    print("\nCleaning pipeline completed successfully.")
