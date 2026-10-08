"""
Visualization package for InsightEngine.
"""

from .charts import (
    PALETTE,
    CLUSTER_COLORS,
    plot_revenue_trend,
    plot_geographic_bar,
    plot_rfm_scatter,
    plot_segment_comparison,
    plot_roc_curves,
    plot_confusion_matrix_heatmap,
    plot_feature_importance,
    plot_correlation_heatmap,
)

__all__ = [
    "PALETTE",
    "CLUSTER_COLORS",
    "plot_revenue_trend",
    "plot_geographic_bar",
    "plot_rfm_scatter",
    "plot_segment_comparison",
    "plot_roc_curves",
    "plot_confusion_matrix_heatmap",
    "plot_feature_importance",
    "plot_correlation_heatmap",
]
