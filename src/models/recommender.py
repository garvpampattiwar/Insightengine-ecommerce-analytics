"""
Collaborative Filtering Product Recommendation Engine.
High-Performance Item-to-Item Nearest Neighbor Architecture.
Uses sparse cosine similarity graphs and compact interaction mappings
to deliver sub-millisecond inference and an ultra-compact artifact footprint (< 2 MB).
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import joblib
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

from ..data.loader import get_project_root


class ItemSimilarityProxy:
    """
    Lightweight, memory-efficient proxy exposing a pandas DataFrame-like interface
    for item-item similarities without storing massive dense matrices in RAM or disk.
    """

    def __init__(self, item_neighbors: Dict[str, List[Tuple[str, float]]], item_list: List[str]):
        self._item_neighbors = item_neighbors
        self._item_list = item_list
        self.columns = pd.Index(item_list)
        self.shape = (len(item_list), len(item_list))

    def __len__(self) -> int:
        return len(self._item_list)

    @property
    def loc(self):
        class _LocIndexer:
            def __init__(self, parent):
                self.parent = parent

            def __getitem__(self, item_code):
                item_str = str(item_code)
                neighbors = self.parent._item_neighbors.get(item_str, [])
                if not neighbors:
                    return pd.Series(dtype=float)
                data = {k: v for k, v in neighbors}
                return pd.Series(data)

        return _LocIndexer(self)


class UserItemProxy:
    """
    Lightweight, memory-efficient proxy exposing user-item interaction matrix
    dimensions and index access without storing a 100+ MB dense 2D float array.
    """

    def __init__(self, user_purchases: Dict[int, Dict[str, float]], item_list: List[str]):
        self._user_purchases = user_purchases
        self._item_list = item_list
        self.index = pd.Index(list(user_purchases.keys()))
        self.columns = pd.Index(item_list)
        self.shape = (len(user_purchases), len(item_list))

    def __len__(self) -> int:
        return len(self._user_purchases)

    @property
    def loc(self):
        class _UserLocIndexer:
            def __init__(self, parent):
                self.parent = parent

            def __getitem__(self, cid):
                try:
                    cid_int = int(cid)
                except (ValueError, TypeError):
                    cid_int = cid
                data = self.parent._user_purchases.get(cid_int, {})
                s = pd.Series(0.0, index=self.parent.columns)
                for k, v in data.items():
                    if k in s.index:
                        s[k] = v
                return s

        return _UserLocIndexer(self)


class CollaborativeRecommender:
    """
    Item-Item Collaborative Filtering Recommendation System.
    Stores compact item nearest-neighbor co-purchase graphs and customer interaction histories.
    """

    def __init__(self, top_n_popular: int = 20, top_k_neighbors: int = 60):
        self.top_n_popular = top_n_popular
        self.top_k_neighbors = top_k_neighbors
        self.product_catalog: Dict[str, str] = {}
        self.popular_products: List[Dict[str, Any]] = []
        self.item_neighbors: Dict[str, List[Tuple[str, float]]] = {}
        self.user_purchases: Dict[int, Dict[str, float]] = {}
        self.item_list: List[str] = []
        self._user_item_proxy: Optional[UserItemProxy] = None
        self._item_similarity_proxy: Optional[ItemSimilarityProxy] = None

    @property
    def user_item_matrix(self) -> UserItemProxy:
        if self._user_item_proxy is None:
            self._user_item_proxy = UserItemProxy(self.user_purchases, self.item_list)
        return self._user_item_proxy

    @property
    def item_similarity_df(self) -> ItemSimilarityProxy:
        if self._item_similarity_proxy is None:
            self._item_similarity_proxy = ItemSimilarityProxy(self.item_neighbors, self.item_list)
        return self._item_similarity_proxy

    @property
    def known_customers(self) -> List[int]:
        return list(self.user_purchases.keys())

    def fit(self, df: pd.DataFrame, min_product_orders: int = 5) -> "CollaborativeRecommender":
        """
        Build compact item-item similarity graph and user purchase history.
        """
        if "Revenue" not in df.columns:
            df = df.copy()
            df["Revenue"] = df["Quantity"] * df["UnitPrice"]

        # Catalog mapping: StockCode -> Description
        cat_map = (
            df.groupby("StockCode")["Description"]
            .agg(lambda s: s.mode().iloc[0] if not s.empty else "Unknown Product")
            .to_dict()
        )
        self.product_catalog = {str(k): str(v) for k, v in cat_map.items()}

        # Filter items with at least min_product_orders for stability
        order_counts = df.groupby("StockCode")["InvoiceNo"].nunique()
        frequent_items = order_counts[order_counts >= min_product_orders].index
        df_filtered = df[df["StockCode"].isin(frequent_items)].copy()

        # Popular products fallback list
        pop_summary = (
            df.groupby("StockCode")
            .agg(
                OrderCount=("InvoiceNo", "nunique"),
                TotalRevenue=("Revenue", "sum"),
                TotalUnits=("Quantity", "sum"),
            )
            .reset_index()
            .sort_values("OrderCount", ascending=False)
        )
        self.popular_products = []
        for _, row in pop_summary.head(self.top_n_popular).iterrows():
            code = str(row["StockCode"])
            self.popular_products.append({
                "StockCode": code,
                "Description": self.product_catalog.get(code, "Popular Item"),
                "OrderCount": int(row["OrderCount"]),
                "TotalRevenue": float(round(row["TotalRevenue"], 2)),
                "Score": 1.0,
                "Reason": "Store bestseller (high customer demand)",
            })

        # Build user-item interactions
        interaction_df = df_filtered.groupby(["CustomerID", "StockCode"])["Quantity"].sum().reset_index()
        interaction_df["Weight"] = np.log1p(np.maximum(interaction_df["Quantity"], 1)).astype(np.float32)

        # Store compact user_purchases dictionary
        self.user_purchases = {}
        for cid, group in interaction_df.groupby("CustomerID"):
            self.user_purchases[int(cid)] = dict(
                zip(group["StockCode"].astype(str), group["Weight"].round(3).astype(float))
            )

        # Build sparse matrix for pairwise similarity
        pivot = interaction_df.pivot(index="CustomerID", columns="StockCode", values="Weight").fillna(0)
        self.item_list = pivot.columns.astype(str).tolist()
        num_items = len(self.item_list)

        # Compute cosine similarity using sparse representations
        item_matrix = pivot.values.T.astype(np.float32)
        sparse_items = csr_matrix(item_matrix)
        sim_matrix = cosine_similarity(sparse_items, dense_output=True).astype(np.float32)

        # Extract only Top-K nearest neighbors per item to eliminate 98% of near-zero weights
        self.item_neighbors = {}
        for i, item_code in enumerate(self.item_list):
            row_sims = sim_matrix[i]
            # Exclude self-similarity at index i
            top_indices = np.argsort(row_sims)[::-1][1 : self.top_k_neighbors + 1]
            self.item_neighbors[item_code] = [
                (self.item_list[idx], round(float(row_sims[idx]), 4))
                for idx in top_indices
                if row_sims[idx] > 0.01
            ]

        # Reset proxies so they rebind
        self._user_item_proxy = None
        self._item_similarity_proxy = None

        return self

    def recommend(
        self,
        customer_id: Any,
        n: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Generate top-n product recommendations for a customer.
        Falls back to popularity baseline if customer is unknown or has no history.
        """
        try:
            cid = int(customer_id)
        except (ValueError, TypeError):
            cid = customer_id

        # Cold-start fallback: Unknown customer or no interaction history
        if cid not in self.user_purchases:
            fallback = self.popular_products[:n]
            return [
                {
                    "StockCode": p["StockCode"],
                    "Description": p["Description"],
                    "Score": 0.50,
                    "Reason": "New/Unknown customer fallback: top trending store bestseller",
                }
                for p in fallback
            ]

        purchased_items = self.user_purchases[cid]
        if not purchased_items:
            fallback = self.popular_products[:n]
            return [
                {
                    "StockCode": p["StockCode"],
                    "Description": p["Description"],
                    "Score": 0.50,
                    "Reason": "Customer with no past history: trending bestseller",
                }
                for p in fallback
            ]

        # Aggregate candidate co-purchase scores across all purchased items
        candidate_scores: Dict[str, float] = {}
        candidate_sim_sums: Dict[str, float] = {}
        best_triggers: Dict[str, Tuple[str, float]] = {}

        for item, weight in purchased_items.items():
            neighbors = self.item_neighbors.get(item, [])
            for neighbor, sim in neighbors:
                if neighbor not in purchased_items:
                    candidate_scores[neighbor] = candidate_scores.get(neighbor, 0.0) + (sim * weight)
                    candidate_sim_sums[neighbor] = candidate_sim_sums.get(neighbor, 0.0) + sim
                    if neighbor not in best_triggers or sim > best_triggers[neighbor][1]:
                        best_triggers[neighbor] = (item, sim)

        if not candidate_scores:
            return [
                {
                    "StockCode": p["StockCode"],
                    "Description": p["Description"],
                    "Score": 0.50,
                    "Reason": "Store bestseller",
                }
                for p in self.popular_products[:n]
            ]

        # Compute normalized candidate scores
        scored_candidates = []
        for cand, score in candidate_scores.items():
            sim_sum = candidate_sim_sums.get(cand, 1.0) + 1e-6
            norm_score = round(score / sim_sum, 4)
            trigger_item = best_triggers[cand][0]
            trigger_name = self.product_catalog.get(trigger_item, "previous purchase")
            scored_candidates.append({
                "StockCode": str(cand),
                "Description": self.product_catalog.get(str(cand), "Product " + str(cand)),
                "Score": float(norm_score),
                "Reason": f"High affinity with your purchase: '{trigger_name[:35]}'",
            })

        # Sort descending by score
        scored_candidates.sort(key=lambda x: x["Score"], reverse=True)
        return scored_candidates[:n]

    def evaluate_coverage(self, sample_size: int = 500, n: int = 5) -> Dict[str, Any]:
        """
        Evaluate catalog coverage and diversity across active customers.
        """
        all_customers = list(self.user_purchases.keys())
        if not all_customers:
            return {"catalog_coverage_pct": 0.0}

        sample_users = (
            all_customers if len(all_customers) <= sample_size
            else np.random.choice(all_customers, size=sample_size, replace=False)
        )

        recommended_items = set()
        for u in sample_users:
            recs = self.recommend(u, n=n)
            for r in recs:
                recommended_items.add(r["StockCode"])

        total_catalog = max(1, len(self.item_list))
        coverage_pct = round((len(recommended_items) / total_catalog) * 100, 2)

        return {
            "users_evaluated": len(sample_users),
            "unique_items_recommended": len(recommended_items),
            "active_catalog_items": total_catalog,
            "catalog_coverage_pct": coverage_pct,
        }

    def save_artifacts(self, output_dir: Optional[Path] = None) -> None:
        """
        Serialize ultra-compact recommender artifacts with compression (< 2 MB).
        """
        root = get_project_root()
        art_dir = output_dir or (root / "models" / "recommender")
        art_dir.mkdir(parents=True, exist_ok=True)

        # Clear proxies before dump to guarantee minimal memory footprint
        self._user_item_proxy = None
        self._item_similarity_proxy = None

        # Save model object using joblib compression level 3
        model_file = art_dir / "recommender_model.pkl"
        joblib.dump(self, model_file, compress=3)

        # Save lightweight JSON artifacts
        with open(art_dir / "product_catalog.json", "w") as f:
            json.dump(self.product_catalog, f, indent=2)

        with open(art_dir / "popular_products.json", "w") as f:
            json.dump(self.popular_products, f, indent=2)

        model_size_mb = model_file.stat().st_size / (1024 * 1024)

        metadata = {
            "num_products": len(self.item_list),
            "num_customers": len(self.user_purchases),
            "top_popular_count": len(self.popular_products),
            "model_size_mb": round(model_size_mb, 2),
            "compression": "joblib_zlib_level_3",
            "architecture": "item_to_item_nearest_neighbor_graph",
        }
        with open(art_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)


def load_recommender_artifacts() -> Dict[str, Any]:
    """Load serialized recommendation engine."""
    root = get_project_root()
    art_dir = root / "models" / "recommender"

    model_file = art_dir / "recommender_model.pkl"
    if not model_file.exists():
        raise FileNotFoundError(f"Recommender artifacts not found at {art_dir}. Train recommender first.")

    model: CollaborativeRecommender = joblib.load(model_file)

    with open(art_dir / "popular_products.json", "r") as f:
        popular = json.load(f)
    with open(art_dir / "product_catalog.json", "r") as f:
        catalog = json.load(f)

    return {
        "model": model,
        "popular_products": popular,
        "product_catalog": catalog,
    }
