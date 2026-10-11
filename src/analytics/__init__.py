"""
Analytics package for InsightEngine: EDA, Statistics, and Business Insights.
"""

from .eda import (
    get_summary_kpis,
    get_time_series_aggregations,
    get_geographic_aggregations,
    get_top_products_summary,
    get_bivariate_correlations,
)
from .statistics import (
    run_ab_test_analysis,
    simulate_marketing_ab_test,
)
from .business_insights import generate_executive_insights
from .precompute import (
    precompute_dashboard_aggregates,
    load_precomputed_aggregates,
    get_precomputed_path,
)

__all__ = [
    "get_summary_kpis",
    "get_time_series_aggregations",
    "get_geographic_aggregations",
    "get_top_products_summary",
    "get_bivariate_correlations",
    "run_ab_test_analysis",
    "simulate_marketing_ab_test",
    "generate_executive_insights",
    "precompute_dashboard_aggregates",
    "load_precomputed_aggregates",
    "get_precomputed_path",
]
