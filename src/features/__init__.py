"""
Feature engineering package for InsightEngine.
"""

from .rfm import calculate_rfm, preprocess_rfm_for_clustering
from .customer_features import (
    create_customer_features,
    create_time_based_churn_dataset,
)
from .product_features import create_product_features

__all__ = [
    "calculate_rfm",
    "preprocess_rfm_for_clustering",
    "create_customer_features",
    "create_time_based_churn_dataset",
    "create_product_features",
]
