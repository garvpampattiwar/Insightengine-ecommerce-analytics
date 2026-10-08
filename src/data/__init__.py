"""
Data loading and validation module.
"""

from .loader import (
    get_project_root,
    load_raw_data,
    load_cleaned_data,
    validate_raw_schema,
    validate_cleaned_schema,
)
from .cleaning import clean_online_retail

__all__ = [
    "get_project_root",
    "load_raw_data",
    "load_cleaned_data",
    "validate_raw_schema",
    "validate_cleaned_schema",
    "clean_online_retail",
]
