"""
End-to-End Model Training and Artifact Generation Pipeline.
Executes data loading, RFM segmentation, churn modeling, and recommender fitting.
Saves all model artifacts to models/ for Streamlit and report generation.
"""

import sys
from pathlib import Path
import json

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.data.loader import load_cleaned_data
from src.features.rfm import calculate_rfm
from src.features.customer_features import create_time_based_churn_dataset
from src.models.segmentation import train_segmentation_model
from src.models.churn import train_churn_models
from src.models.recommender import CollaborativeRecommender
from src.analytics.statistics import simulate_marketing_ab_test
from src.analytics.business_insights import generate_executive_insights
from src.analytics.precompute import precompute_dashboard_aggregates


def run_pipeline():
    print("=" * 60)
    print("INSIGHTENGINE: END-TO-END MODEL TRAINING PIPELINE")
    print("=" * 60)

    # 1. Load Cleaned Dataset
    print("\n[Step 1/5] Loading cleaned e-commerce dataset...")
    df = load_cleaned_data()
    print(f"Loaded {len(df):,} transactions across {df['CustomerID'].nunique():,} unique customers.")

    # 2. RFM & Customer Segmentation
    print("\n[Step 2/5] Performing RFM analysis and K-Means segmentation...")
    rfm_df = calculate_rfm(df)
    kmeans, scaler, clustered_rfm, profiles, eval_seg = train_segmentation_model(
        rfm_df, n_clusters=4, random_state=42, save_artifacts=True
    )
    print("Segmentation Model trained and saved:")
    for _, row in profiles.iterrows():
        print(f"  - {row['SegmentName']}: {row['CustomerCount']} customers ({row['CustomerPct']}%), ${row['TotalRevenue']:,.2f} rev ({row['RevenuePct']}%)")

    # 3. Churn Modeling
    print("\n[Step 3/5] Building time-aware churn dataset and training ML models...")
    churn_df, cutoff_date, max_date = create_time_based_churn_dataset(df, observation_days=90)
    print(f"Observation Cutoff Date: {cutoff_date.date()} | Max Date: {max_date.date()}")
    print(f"Evaluated customers: {len(churn_df):,} | Churn rate: {churn_df['Churn'].mean():.2%}")

    churn_results = train_churn_models(churn_df, save_artifacts=True)
    best_m = churn_results["best_model_name"]
    best_auc = churn_results["metrics"][best_m]["roc_auc"]
    best_f1 = churn_results["metrics"][best_m]["f1"]
    print(f"Churn Models trained. Best Model: {best_m.upper()} (ROC-AUC = {best_auc:.4f}, F1 = {best_f1:.4f})")
    print(f"LR Metrics: {churn_results['metrics']['logistic_regression']}")
    print(f"RF Metrics: {churn_results['metrics']['random_forest']}")

    # 4. Collaborative Filtering Recommender
    print("\n[Step 4/5] Fitting Collaborative Filtering Recommendation Engine...")
    rec_engine = CollaborativeRecommender(top_n_popular=20)
    rec_engine.fit(df, min_product_orders=5)
    coverage = rec_engine.evaluate_coverage(sample_size=300, n=5)
    rec_engine.save_artifacts()
    print(f"Recommender trained and saved. Catalog coverage: {coverage['catalog_coverage_pct']}% across sample users.")
    sample_cid = df["CustomerID"].iloc[0]
    sample_recs = rec_engine.recommend(sample_cid, n=3)
    print(f"Sample recommendations for Customer {sample_cid}:")
    for r in sample_recs:
        print(f"  * [{r['StockCode']}] {r['Description']} (Score: {r['Score']:.4f})")

    # 5. Statistical Testing & Executive Insights
    print("\n[Step 5/5] Running A/B test simulation and computing executive insights...")
    ab_results = simulate_marketing_ab_test(df, treatment_lift_pct=7.5, random_state=42)
    print(f"A/B Test Result: {ab_results['decision_summary']}")
    print(f"p-value: {ab_results['t_test']['p_value']:.4e} | Cohen's d: {ab_results['effect_size_cohens_d']}")

    insights = generate_executive_insights(df, segmented_df=clustered_rfm, churn_df=churn_results["scored_df"])
    precompute_dashboard_aggregates(df, save=True)
    
    # Save pipeline summary
    summary_path = root_dir / "reports" / "pipeline_run_summary.json"
    summary_data = {
        "dataset_transactions": len(df),
        "total_revenue": float(round(df["Revenue"].sum(), 2)),
        "unique_customers": int(df["CustomerID"].nunique()),
        "unique_products": int(df["StockCode"].nunique()),
        "segmentation_profiles": profiles.to_dict(orient="records"),
        "churn_metrics": churn_results["metrics"],
        "churn_risk_counts": churn_results["risk_counts"],
        "best_churn_model": best_m,
        "recommender_coverage": coverage,
        "ab_test": {
            "difference_in_means": ab_results["difference_in_means"],
            "percentage_lift": ab_results["percentage_lift"],
            "t_p_value": ab_results["t_test"]["p_value"],
            "mwu_p_value": ab_results["mann_whitney_u"]["p_value"],
            "cohens_d": ab_results["effect_size_cohens_d"],
            "is_significant": ab_results["is_statistically_significant"],
        },
        "executive_insights": insights,
    }
    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"Pipeline summary exported to {summary_path}")
    print("\n" + "=" * 60)
    print("ALL MODELS AND ARTIFACTS SUCCESSFULLY GENERATED!")
    print("=" * 60)


if __name__ == "__main__":
    run_pipeline()
