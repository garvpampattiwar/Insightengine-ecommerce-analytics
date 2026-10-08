"""
Unit tests for data loader and data cleaning pipeline.
"""

import pytest
import pandas as pd
import numpy as np

from src.data.loader import validate_raw_schema, validate_cleaned_schema, load_cleaned_data
from src.data.cleaning import clean_online_retail


@pytest.fixture
def sample_raw_data():
    return pd.DataFrame({
        "InvoiceNo": ["536365", "C536366", "536367", "536368", "536369", "536370", "536365"],
        "StockCode": ["85123A", "85123A", "71053", "84406B", "POST", "22423", "85123A"],
        "Description": ["WHITE HANGING HEART", "CANCELLATION", "WHITE METAL LANTERN", "CREAM CUPID", "POSTAGE", "REGENCY CAKESTAND", "WHITE HANGING HEART"],
        "Quantity": [6, -1, 2, 5, 1, -3, 6],
        "InvoiceDate": pd.to_datetime(["2011-01-01 10:00:00"] * 7),
        "UnitPrice": [2.55, 2.55, 3.39, 4.25, 18.0, 12.75, 2.55],
        "CustomerID": [17850.0, 17850.0, 13047.0, np.nan, 12583.0, 14000.0, 17850.0],
        "Country": ["United Kingdom", "United Kingdom", "United Kingdom", "United Kingdom", "France", "United Kingdom", "United Kingdom"],
    })


def test_validate_raw_schema(sample_raw_data):
    is_valid, missing = validate_raw_schema(sample_raw_data)
    assert is_valid is True
    assert len(missing) == 0

    invalid_df = sample_raw_data.drop(columns=["InvoiceNo"])
    is_valid_inv, missing_inv = validate_raw_schema(invalid_df)
    assert is_valid_inv is False
    assert "InvoiceNo" in missing_inv


def test_clean_online_retail(sample_raw_data):
    cleaned, audit = clean_online_retail(sample_raw_data, save_interim=False, save_processed=False)
    
    # Assert duplicates were dropped (1 duplicate row in fixture)
    assert audit["duplicates_removed"] == 1
    # Assert cancellation with 'C' was removed
    assert audit["cancellations_separated"] == 1
    # Assert missing CustomerID was dropped
    assert audit["missing_customer_id_removed"] == 1
    # Assert administrative code 'POST' was removed
    assert audit["admin_codes_removed"] == 1
    
    # Assert Revenue column was created correctly: Quantity * UnitPrice
    assert "Revenue" in cleaned.columns
    assert (cleaned["Revenue"] == np.round(cleaned["Quantity"] * cleaned["UnitPrice"], 2)).all()
    assert (cleaned["Quantity"] > 0).all()
    assert (cleaned["UnitPrice"] > 0).all()
    assert not cleaned["CustomerID"].isnull().any()
    assert issubclass(cleaned["CustomerID"].dtype.type, np.integer)


def test_load_cleaned_data_exists():
    cleaned_df = load_cleaned_data()
    assert not cleaned_df.empty
    assert len(cleaned_df) > 100000
    is_valid, missing = validate_cleaned_schema(cleaned_df)
    assert is_valid is True
    assert len(missing) == 0
