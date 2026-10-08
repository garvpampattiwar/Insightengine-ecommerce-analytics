"""
Models package for InsightEngine: Customer Segmentation, Churn Prediction, and Recommender.
"""

from .segmentation import (
    train_segmentation_model,
    evaluate_kmeans_clusters,
    load_segmentation_artifacts,
)
from .churn import (
    train_churn_models,
    load_churn_artifacts,
    FEATURE_COLUMNS,
)
from .recommender import (
    CollaborativeRecommender,
    load_recommender_artifacts,
)

__all__ = [
    "train_segmentation_model",
    "evaluate_kmeans_clusters",
    "load_segmentation_artifacts",
    "train_churn_models",
    "load_churn_artifacts",
    "FEATURE_COLUMNS",
    "CollaborativeRecommender",
    "load_recommender_artifacts",
]
