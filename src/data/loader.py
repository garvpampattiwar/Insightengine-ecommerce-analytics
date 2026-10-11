"""
Data loading and schema validation utilities for InsightEngine.
"""

import os
from pathlib import Path
from typing import Optional, Tuple, List
import pandas as pd


def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return Path(__file__).resolve().parent.parent.parent


def validate_raw_schema(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate that the input DataFrame matches expected raw Online Retail schema.
    
    Returns:
        (is_valid, missing_columns)
    """
    required_cols = [
        "InvoiceNo",
        "StockCode",
        "Description",
        "Quantity",
        "InvoiceDate",
        "UnitPrice",
        "CustomerID",
        "Country",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    return len(missing) == 0, missing


def validate_cleaned_schema(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate that the cleaned DataFrame has the necessary fields.
    """
    required_cols = [
        "InvoiceNo",
        "StockCode",
        "Description",
        "Quantity",
        "InvoiceDate",
        "UnitPrice",
        "CustomerID",
        "Country",
        "Revenue",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    return len(missing) == 0, missing


def load_raw_data(data_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load raw Online Retail dataset.
    Prioritizes fast CSV if cached in data/raw, falls back to Excel, 
    and raises actionable FileNotFoundError if neither exists.
    """
    root = get_project_root()
    
    if data_path:
        path = Path(data_path)
    else:
        csv_candidate = root / "data" / "raw" / "online_retail_raw.csv"
        xlsx_candidate = root / "data" / "raw" / "Online Retail.xlsx"
        
        if csv_candidate.exists():
            path = csv_candidate
        elif xlsx_candidate.exists():
            path = xlsx_candidate
        else:
            raise FileNotFoundError(
                f"No raw dataset found in {root / 'data' / 'raw'}.\n"
                f"Please place 'Online Retail.xlsx' or 'online_retail_raw.csv' in the data/raw directory, "
                f"or run the download script."
            )
            
    if str(path).endswith(".csv"):
        df = pd.read_csv(
            path,
            dtype={"InvoiceNo": str, "StockCode": str, "Description": str, "Country": str},
            parse_dates=["InvoiceDate"]
        )
    elif str(path).endswith(".xlsx") or str(path).endswith(".xls"):
        df = pd.read_excel(path)
        if "InvoiceDate" in df.columns:
            df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    else:
        raise ValueError(f"Unsupported file format: {path}")

    is_valid, missing = validate_raw_schema(df)
    if not is_valid:
        raise ValueError(f"Raw dataset is missing required columns: {missing}")

    return df


def load_cleaned_data(data_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load cleaned and preprocessed e-commerce dataset from data/processed/.
    Prioritizes ultra-fast parquet format if available, falling back to CSV.
    """
    root = get_project_root()
    if data_path:
        path = Path(data_path)
    else:
        parquet_candidate = root / "data" / "processed" / "cleaned_ecommerce.parquet"
        csv_candidate = root / "data" / "processed" / "cleaned_ecommerce.csv"
        path = parquet_candidate if parquet_candidate.exists() else csv_candidate

    if not path.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found at '{path}'.\n"
            f"Please run the data cleaning pipeline: python -m src.data.cleaning or run notebooks/01_data_cleaning.ipynb."
        )

    if str(path).endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(
            path,
            dtype={
                "InvoiceNo": str,
                "StockCode": str,
                "Description": str,
                "CustomerID": int,
                "Country": str,
            },
            parse_dates=["InvoiceDate"],
        )

    is_valid, missing = validate_cleaned_schema(df)
    if not is_valid:
        raise ValueError(f"Cleaned dataset is missing required columns: {missing}")

    return df
