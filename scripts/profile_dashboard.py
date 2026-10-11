import time
import sys
from pathlib import Path
import pandas as pd
import numpy as np

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

def benchmark():
    timings = {}

    # 1. Load cleaned data from CSV
    t0 = time.perf_counter()
    from src.data.loader import load_cleaned_data
    df = load_cleaned_data()
    timings["load_cleaned_data"] = time.perf_counter() - t0

    # 2. Add derived time columns (as done in get_cached_data)
    t0 = time.perf_counter()
    df_with_cols = df.copy()
    if "Date" not in df_with_cols.columns:
        df_with_cols["Date"] = df_with_cols["InvoiceDate"].dt.date
    if "YearMonth" not in df_with_cols.columns:
        df_with_cols["YearMonth"] = df_with_cols["InvoiceDate"].dt.strftime("%Y-%m")
    if "DayOfWeek" not in df_with_cols.columns:
        df_with_cols["DayOfWeek"] = df_with_cols["InvoiceDate"].dt.day_name()
    if "Hour" not in df_with_cols.columns:
        df_with_cols["Hour"] = df_with_cols["InvoiceDate"].dt.hour
    timings["derive_datetime_cols"] = time.perf_counter() - t0

    # 3. Load artifacts
    t0 = time.perf_counter()
    from src.models.segmentation import load_segmentation_artifacts
    seg_art = load_segmentation_artifacts()
    timings["load_segmentation_artifacts"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    from src.models.churn import load_churn_artifacts
    churn_art = load_churn_artifacts()
    timings["load_churn_artifacts"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    from src.models.recommender import load_recommender_artifacts
    rec_art = load_recommender_artifacts()
    timings["load_recommender_artifacts"] = time.perf_counter() - t0

    # 4. Analytics aggregations
    from src.analytics.eda import (
        get_summary_kpis,
        get_time_series_aggregations,
        get_geographic_aggregations,
        get_top_products_summary,
        get_bivariate_correlations,
    )
    from src.analytics.business_insights import generate_executive_insights

    t0 = time.perf_counter()
    kpis = get_summary_kpis(df_with_cols)
    timings["get_summary_kpis"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    ts = get_time_series_aggregations(df_with_cols)
    timings["get_time_series_aggregations"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    geo = get_geographic_aggregations(df_with_cols)
    timings["get_geographic_aggregations"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    exec_ins = generate_executive_insights(
        df_with_cols,
        segmented_df=seg_art["customer_segments"],
        churn_df=churn_art["customer_churn_scored"],
    )
    timings["generate_executive_insights"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    top_p = get_top_products_summary(df_with_cols, top_n=10)
    timings["get_top_products_summary"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    corr = get_bivariate_correlations(churn_art["customer_churn_scored"])
    timings["get_bivariate_correlations"] = time.perf_counter() - t0

    # 5. Order values groupby
    t0 = time.perf_counter()
    order_vals = df_with_cols.groupby("InvoiceNo")["Revenue"].sum().values
    timings["order_values_groupby"] = time.perf_counter() - t0

    # 6. Recommender inference
    rec_model = rec_art["model"]
    t0 = time.perf_counter()
    recs_known = rec_model.recommend(rec_model.known_customers[0], n=5)
    timings["rec_known_customer"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    recs_unknown = rec_model.recommend("99999999", n=5)
    timings["rec_unknown_customer"] = time.perf_counter() - t0

    print("=== PROFILING RESULTS ===")
    total = 0
    for k, v in timings.items():
        print(f"{k:35s}: {v*1000:8.2f} ms")
        total += v
    print(f"{'TOTAL MEASURED TIME':35s}: {total*1000:8.2f} ms ({total:.2f} s)")

if __name__ == "__main__":
    benchmark()
