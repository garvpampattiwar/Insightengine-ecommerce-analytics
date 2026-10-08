"""
Customer Segmentation Model using RFM and K-Means Clustering.
Includes optimal-k evaluation (Elbow and Silhouette), profiling, and data-driven segment naming.
"""

import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
import joblib
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from ..data.loader import get_project_root
from ..features.rfm import preprocess_rfm_for_clustering


def evaluate_kmeans_clusters(
    scaled_features: np.ndarray,
    k_range=range(2, 9),
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Evaluate multiple values of K using Inertia (Elbow) and Silhouette Score.
    """
    inertias = []
    silhouette_scores = []
    k_values = list(k_range)

    for k in k_values:
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=random_state)
        cluster_labels = km.fit_predict(scaled_features)
        inertias.append(float(km.inertia_))
        sil = float(silhouette_score(scaled_features, cluster_labels))
        silhouette_scores.append(sil)

    best_k_idx = int(np.argmax(silhouette_scores))
    best_k = k_values[best_k_idx]

    return {
        "k_values": k_values,
        "inertias": inertias,
        "silhouette_scores": silhouette_scores,
        "recommended_k_silhouette": best_k,
    }


def assign_segment_names(profile_df: pd.DataFrame) -> Dict[int, str]:
    """
    Assign meaningful business names based on actual cluster characteristics.
    Sorts clusters by monetary value and recency to map to:
    - Champions
    - Loyal Customers
    - Potential / Developing
    - At-Risk / Hibernating
    """
    # Rank clusters:
    # High monetary + high frequency + low recency is top tier
    # Low monetary + low frequency + high recency is churned/hibernating
    df = profile_df.copy()
    
    # Calculate a composite health score: higher M and F, lower R
    # Normalize rank across clusters
    m_rank = df["Monetary_mean"].rank(ascending=True)
    f_rank = df["Frequency_mean"].rank(ascending=True)
    r_rank = df["Recency_mean"].rank(ascending=False) # Lower recency gets higher score
    
    composite = m_rank + f_rank + r_rank
    sorted_clusters = composite.sort_values(ascending=False).index.tolist()

    names = {}
    if len(sorted_clusters) == 4:
        names[sorted_clusters[0]] = "Champions"
        names[sorted_clusters[1]] = "Loyal Customers"
        names[sorted_clusters[2]] = "Potential Loyalists"
        names[sorted_clusters[3]] = "At Risk / Hibernating"
    elif len(sorted_clusters) == 3:
        names[sorted_clusters[0]] = "High-Value Champions"
        names[sorted_clusters[1]] = "Regular Core"
        names[sorted_clusters[2]] = "Dormant / At-Risk"
    elif len(sorted_clusters) == 5:
        names[sorted_clusters[0]] = "Champions"
        names[sorted_clusters[1]] = "Loyal Customers"
        names[sorted_clusters[2]] = "Potential Loyalists"
        names[sorted_clusters[3]] = "At Risk"
        names[sorted_clusters[4]] = "Lost / Hibernating"
    else:
        for i, c in enumerate(sorted_clusters):
            names[c] = f"Segment Tier {i + 1}"

    return names


def get_segment_recommendation(segment_name: str) -> str:
    """Provide tailored marketing and retention strategies for each segment."""
    recs = {
        "Champions": "VIP loyalty programs, early access to new collections, personalized white-glove service, referral incentives.",
        "Loyal Customers": "Cross-sell high-affinity accessories, loyalty point accelerators, premium tier incentives to increase basket size.",
        "Potential Loyalists": "Nurturing email flows, limited-time trial discounts on related categories, low-friction checkout reminders.",
        "At Risk": "Proactive re-engagement campaigns, win-back surveys, time-sensitive personalized discount vouchers.",
        "At Risk / Hibernating": "Automated re-engagement discount, brand reminder emails, survey reason for drop-off before sunsetting.",
        "Lost / Hibernating": "Low-frequency reactivation campaigns; avoid ad spend unless on high-intent seasonal clearance.",
    }
    return recs.get(segment_name, "Engage with personalized recommendations and monitor lifetime value.")


def train_segmentation_model(
    rfm_df: pd.DataFrame,
    n_clusters: int = 4,
    random_state: int = 42,
    save_artifacts: bool = True,
) -> Tuple[KMeans, Any, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Train K-Means segmentation model and compute segment profiles.

    Returns:
        (kmeans_model, scaler, clustered_rfm_df, profile_df, evaluation_results)
    """
    root = get_project_root()
    scaled_data, scaler, _ = preprocess_rfm_for_clustering(rfm_df)

    eval_results = evaluate_kmeans_clusters(scaled_data, k_range=range(2, 8), random_state=random_state)

    # Fit final K-Means
    kmeans = KMeans(n_clusters=n_clusters, init="k-means++", n_init=10, random_state=random_state)
    labels = kmeans.fit_predict(scaled_data)

    df_clustered = rfm_df.copy()
    df_clustered["Cluster"] = labels

    # Build profiles
    profile_raw = df_clustered.groupby("Cluster").agg(
        Recency_mean=("Recency", "mean"),
        Recency_median=("Recency", "median"),
        Frequency_mean=("Frequency", "mean"),
        Frequency_median=("Frequency", "median"),
        Monetary_mean=("Monetary", "mean"),
        Monetary_median=("Monetary", "median"),
        Monetary_sum=("Monetary", "sum"),
        CustomerCount=("CustomerID", "count"),
    )

    cluster_names = assign_segment_names(profile_raw)
    df_clustered["SegmentName"] = df_clustered["Cluster"].map(cluster_names)

    total_customers = len(df_clustered)
    total_rev = df_clustered["Monetary"].sum()

    profile_df = df_clustered.groupby(["Cluster", "SegmentName"]).agg(
        CustomerCount=("CustomerID", "count"),
        AvgRecency=("Recency", "mean"),
        MedianRecency=("Recency", "median"),
        AvgFrequency=("Frequency", "mean"),
        MedianFrequency=("Frequency", "median"),
        AvgMonetary=("Monetary", "mean"),
        MedianMonetary=("Monetary", "median"),
        TotalRevenue=("Monetary", "sum"),
    ).reset_index()

    profile_df["CustomerPct"] = np.round((profile_df["CustomerCount"] / total_customers) * 100, 2)
    profile_df["RevenuePct"] = np.round((profile_df["TotalRevenue"] / total_rev) * 100, 2)
    profile_df["AvgRecency"] = np.round(profile_df["AvgRecency"], 1)
    profile_df["AvgFrequency"] = np.round(profile_df["AvgFrequency"], 1)
    profile_df["AvgMonetary"] = np.round(profile_df["AvgMonetary"], 2)
    profile_df["TotalRevenue"] = np.round(profile_df["TotalRevenue"], 2)
    profile_df["RecommendedAction"] = profile_df["SegmentName"].apply(get_segment_recommendation)

    if save_artifacts:
        art_dir = root / "models" / "segmentation"
        art_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(kmeans, art_dir / "kmeans.pkl")
        joblib.dump(scaler, art_dir / "scaler.pkl")

        metadata = {
            "n_clusters": n_clusters,
            "random_state": random_state,
            "cluster_names": {int(k): v for k, v in cluster_names.items()},
            "evaluation": eval_results,
            "total_customers": total_customers,
            "total_revenue": float(total_rev),
        }
        with open(art_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        profile_df.to_csv(art_dir / "segment_profiles.csv", index=False)
        df_clustered.to_csv(art_dir / "customer_segments.csv", index=False)

    return kmeans, scaler, df_clustered, profile_df, eval_results


def load_segmentation_artifacts() -> Dict[str, Any]:
    """Load serialized segmentation models and metadata."""
    root = get_project_root()
    art_dir = root / "models" / "segmentation"
    
    if not (art_dir / "kmeans.pkl").exists():
        raise FileNotFoundError(f"Segmentation artifacts not found at {art_dir}. Please train segmentation first.")

    kmeans = joblib.load(art_dir / "kmeans.pkl")
    scaler = joblib.load(art_dir / "scaler.pkl")
    with open(art_dir / "metadata.json", "r") as f:
        metadata = json.load(f)
    
    profiles = pd.read_csv(art_dir / "segment_profiles.csv")
    segments = pd.read_csv(art_dir / "customer_segments.csv")

    return {
        "model": kmeans,
        "scaler": scaler,
        "metadata": metadata,
        "profiles": profiles,
        "customer_segments": segments,
    }
