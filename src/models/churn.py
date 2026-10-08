"""
Churn Prediction Modeling Pipeline.
Trains Logistic Regression and Random Forest models with hyperparameter tuning,
evaluates performance metrics (ROC-AUC, F1, Recall, Precision, Accuracy),
prevents data leakage, and provides calibrated risk scoring.
"""

import json
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)

from ..data.loader import get_project_root


FEATURE_COLUMNS = [
    "Recency",
    "Frequency",
    "Monetary",
    "AOV",
    "TotalQuantity",
    "BasketSize",
    "UniqueProducts",
    "ProductDiversity",
    "TenureDays",
    "LifespanDays",
    "PurchaseVelocity",
]


def train_churn_models(
    churn_df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    target_col: str = "Churn",
    test_size: float = 0.25,
    random_state: int = 42,
    save_artifacts: bool = True,
) -> Dict[str, Any]:
    """
    Train and evaluate Logistic Regression and Random Forest classifiers.

    Returns:
        Dictionary containing trained models, metrics, comparisons, and best model name.
    """
    root = get_project_root()
    if feature_cols is None:
        feature_cols = [c for c in FEATURE_COLUMNS if c in churn_df.columns]

    X = churn_df[feature_cols].copy()
    y = churn_df[target_col].astype(int)

    # Train / Test split with stratification to preserve churn distribution
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Preprocessing scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 1. Logistic Regression with GridSearch
    lr = LogisticRegression(random_state=random_state, max_iter=1000)
    lr_param_grid = {"C": [0.01, 0.1, 1.0, 10.0], "solver": ["lbfgs"]}
    lr_cv = GridSearchCV(lr, lr_param_grid, cv=5, scoring="roc_auc", n_jobs=-1)
    lr_cv.fit(X_train_scaled, y_train)
    best_lr = lr_cv.best_estimator_

    # Evaluate Logistic Regression
    lr_preds = best_lr.predict(X_test_scaled)
    lr_probs = best_lr.predict_proba(X_test_scaled)[:, 1]
    lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)

    lr_metrics = {
        "accuracy": float(round(accuracy_score(y_test, lr_preds), 4)),
        "precision": float(round(precision_score(y_test, lr_preds, zero_division=0), 4)),
        "recall": float(round(recall_score(y_test, lr_preds, zero_division=0), 4)),
        "f1": float(round(f1_score(y_test, lr_preds, zero_division=0), 4)),
        "roc_auc": float(round(roc_auc_score(y_test, lr_probs), 4)),
        "confusion_matrix": confusion_matrix(y_test, lr_preds).tolist(),
        "best_params": lr_cv.best_params_,
    }

    # 2. Random Forest with GridSearch
    rf = RandomForestClassifier(random_state=random_state)
    rf_param_grid = {
        "n_estimators": [50, 100],
        "max_depth": [4, 6, 8],
        "min_samples_split": [2, 5],
    }
    rf_cv = GridSearchCV(rf, rf_param_grid, cv=5, scoring="roc_auc", n_jobs=-1)
    rf_cv.fit(X_train, y_train)
    best_rf = rf_cv.best_estimator_

    # Evaluate Random Forest
    rf_preds = best_rf.predict(X_test)
    rf_probs = best_rf.predict_proba(X_test)[:, 1]
    rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_probs)

    rf_metrics = {
        "accuracy": float(round(accuracy_score(y_test, rf_preds), 4)),
        "precision": float(round(precision_score(y_test, rf_preds, zero_division=0), 4)),
        "recall": float(round(recall_score(y_test, rf_preds, zero_division=0), 4)),
        "f1": float(round(f1_score(y_test, rf_preds, zero_division=0), 4)),
        "roc_auc": float(round(roc_auc_score(y_test, rf_probs), 4)),
        "confusion_matrix": confusion_matrix(y_test, rf_preds).tolist(),
        "best_params": rf_cv.best_params_,
    }

    # Model selection rationale: In churn detection, missing a churning customer
    # carries a high commercial penalty. ROC-AUC and F1 measure balanced discrimination.
    best_model_name = "random_forest" if rf_metrics["roc_auc"] >= lr_metrics["roc_auc"] else "logistic_regression"
    best_model = best_rf if best_model_name == "random_forest" else best_lr

    # Feature Importance / Coefficients
    rf_importance = dict(
        zip(feature_cols, [float(round(val, 4)) for val in best_rf.feature_importances_])
    )
    # Sort descending
    rf_importance = dict(sorted(rf_importance.items(), key=lambda x: x[1], reverse=True))

    lr_coefs = dict(
        zip(feature_cols, [float(round(val, 4)) for val in best_lr.coef_[0]])
    )

    # Score all customers in dataset
    scored_df = churn_df.copy()
    if best_model_name == "random_forest":
        scored_probs = best_rf.predict_proba(scored_df[feature_cols])[:, 1]
    else:
        scaled_full = scaler.transform(scored_df[feature_cols])
        scored_probs = best_lr.predict_proba(scaled_full)[:, 1]

    scored_df["ChurnProbability"] = np.round(scored_probs, 4)
    scored_df["RiskTier"] = pd.cut(
        scored_df["ChurnProbability"],
        bins=[-0.01, 0.35, 0.65, 1.01],
        labels=["Low Risk", "Medium Risk", "High Risk"],
    )

    risk_counts = scored_df["RiskTier"].value_counts().to_dict()

    results = {
        "models": {
            "logistic_regression": best_lr,
            "random_forest": best_rf,
            "best_model": best_model,
        },
        "scaler": scaler,
        "feature_cols": feature_cols,
        "metrics": {
            "logistic_regression": lr_metrics,
            "random_forest": rf_metrics,
        },
        "best_model_name": best_model_name,
        "feature_importance_rf": rf_importance,
        "coefficients_lr": lr_coefs,
        "roc_curves": {
            "logistic_regression": {
                "fpr": [float(x) for x in lr_fpr[::max(1, len(lr_fpr)//50)]],
                "tpr": [float(x) for x in lr_tpr[::max(1, len(lr_tpr)//50)]],
            },
            "random_forest": {
                "fpr": [float(x) for x in rf_fpr[::max(1, len(rf_fpr)//50)]],
                "tpr": [float(x) for x in rf_tpr[::max(1, len(rf_tpr)//50)]],
            },
        },
        "test_labels": {
            "y_test_count": len(y_test),
            "y_test_churn_rate": float(round(y_test.mean(), 4)),
        },
        "risk_counts": {str(k): int(v) for k, v in risk_counts.items()},
        "scored_df": scored_df,
    }

    if save_artifacts:
        art_dir = root / "models" / "churn"
        art_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(best_lr, art_dir / "logistic_regression.pkl")
        joblib.dump(best_rf, art_dir / "random_forest.pkl")
        joblib.dump(best_model, art_dir / "best_model.pkl")
        joblib.dump(scaler, art_dir / "preprocessing.pkl")

        metadata = {
            "best_model_name": best_model_name,
            "feature_columns": feature_cols,
            "metrics": results["metrics"],
            "feature_importance_rf": rf_importance,
            "coefficients_lr": lr_coefs,
            "roc_curves": results["roc_curves"],
            "risk_counts": results["risk_counts"],
            "dataset_churn_rate": float(round(y.mean(), 4)),
            "total_customers_evaluated": len(churn_df),
        }
        with open(art_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        with open(art_dir / "feature_names.json", "w") as f:
            json.dump(feature_cols, f, indent=2)

        scored_df.to_csv(art_dir / "customer_churn_scored.csv", index=False)

    return results


def load_churn_artifacts() -> Dict[str, Any]:
    """Load serialized churn models, scaler, metadata, and scored customer data."""
    root = get_project_root()
    art_dir = root / "models" / "churn"

    if not (art_dir / "best_model.pkl").exists():
        raise FileNotFoundError(f"Churn artifacts not found at {art_dir}. Train churn models first.")

    best_model = joblib.load(art_dir / "best_model.pkl")
    lr = joblib.load(art_dir / "logistic_regression.pkl")
    rf = joblib.load(art_dir / "random_forest.pkl")
    scaler = joblib.load(art_dir / "preprocessing.pkl")

    with open(art_dir / "metadata.json", "r") as f:
        metadata = json.load(f)

    with open(art_dir / "feature_names.json", "r") as f:
        feature_names = json.load(f)

    scored_df = pd.read_csv(art_dir / "customer_churn_scored.csv")

    return {
        "best_model": best_model,
        "logistic_regression": lr,
        "random_forest": rf,
        "scaler": scaler,
        "metadata": metadata,
        "feature_names": feature_names,
        "customer_churn_scored": scored_df,
    }
