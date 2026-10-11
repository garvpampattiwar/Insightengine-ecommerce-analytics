"""
Interactive Data Visualization utilities using Plotly.
Provides charts for EDA, RFM, Segmentation, Churn Intelligence, and Recommender.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# Brand Color Palette
PALETTE = {
    "primary": "#4F46E5",     # Indigo
    "secondary": "#06B6D4",   # Cyan
    "accent": "#F59E0B",      # Amber
    "danger": "#EF4444",      # Rose / Red
    "success": "#10B981",     # Emerald
    "purple": "#8B5CF6",      # Purple
    "background": "#0F172A",  # Dark Slate
    "card_bg": "#1E293B",     # Surface Slate
    "text": "#F8FAFC",        # Crisp White
}

CLUSTER_COLORS = {
    "Champions": "#10B981",              # Emerald
    "Loyal Customers": "#3B82F6",        # Blue
    "Potential Loyalists": "#F59E0B",    # Amber
    "At Risk / Hibernating": "#EF4444",  # Red
    "At Risk": "#F97316",                # Orange
    "Lost / Hibernating": "#6B7280",     # Gray
}


def plot_revenue_trend(df: pd.DataFrame, x_col: str = "Date", y_col: str = "Revenue") -> go.Figure:
    """Create interactive revenue trend area/line chart."""
    fig = px.area(
        df,
        x=x_col,
        y=y_col,
        title=f"Trend Over Time: {y_col}",
        labels={x_col: "Timeline", y_col: f"Total {y_col} ($)"},
        color_discrete_sequence=[PALETTE["primary"]],
    )
    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=75, r=25, t=50, b=45),
        hovermode="x unified",
        title=dict(x=0.02, xanchor="left"),
        yaxis=dict(title=f"Total {y_col} ($)"),
    )
    fig.update_traces(line=dict(width=2.5), fillcolor="rgba(79, 70, 229, 0.25)")
    return fig


def plot_geographic_bar(geo_df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """Horizontal bar chart for top revenue-generating countries."""
    sub = geo_df.head(top_n).sort_values("Revenue", ascending=True)
    max_rev = sub["Revenue"].max() if not sub.empty else 100
    fig = px.bar(
        sub,
        x="Revenue",
        y="Country",
        orientation="h",
        text="Revenue",
        title=f"Top {top_n} Geographic Markets by Revenue",
        color="Revenue",
        color_continuous_scale="Purples",
    )
    fig.update_traces(texttemplate="$%{text:,.0f}", textposition="outside", cliponaxis=False)
    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=115, r=85, t=50, b=40),
        coloraxis_showscale=False,
        title=dict(x=0.02, xanchor="left"),
        xaxis=dict(range=[0, max_rev * 1.25], title="Revenue ($)"),
        yaxis=dict(title=""),
    )
    return fig


def plot_rfm_scatter(clustered_df: pd.DataFrame) -> go.Figure:
    """Scatter plot of Recency vs Monetary, colored by Customer Segment."""
    fig = px.scatter(
        clustered_df,
        x="Recency",
        y="Monetary",
        color="SegmentName",
        color_discrete_map=CLUSTER_COLORS,
        hover_data=["CustomerID", "Frequency", "AOV"],
        log_y=True,
        title="Customer Segments: Recency vs Monetary (Log Scale)",
        labels={"Recency": "Recency (Days Inactive)", "Monetary": "Monetary Spend ($)"},
    )
    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=75, r=25, t=50, b=85),
        title=dict(x=0.02, xanchor="left"),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.22,
            xanchor="center",
            x=0.5,
            title_text="",
        ),
        yaxis=dict(dtick=1, title="Monetary Spend ($)"),
    )
    fig.update_traces(marker=dict(size=6, opacity=0.75))
    return fig


def plot_segment_comparison(profile_df: pd.DataFrame) -> go.Figure:
    """Bar chart comparing customer share vs revenue contribution per segment."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=profile_df["SegmentName"],
        y=profile_df["CustomerPct"],
        name="% of Total Customers",
        marker_color=PALETTE["secondary"],
        text=profile_df["CustomerPct"].apply(lambda v: f"{v:.1f}%"),
        textposition="auto",
    ))
    fig.add_trace(go.Bar(
        x=profile_df["SegmentName"],
        y=profile_df["RevenuePct"],
        name="% of Total Revenue",
        marker_color=PALETTE["primary"],
        text=profile_df["RevenuePct"].apply(lambda v: f"{v:.1f}%"),
        textposition="auto",
    ))
    max_pct = max(profile_df["RevenuePct"].max(), profile_df["CustomerPct"].max()) if not profile_df.empty else 100
    fig.update_layout(
        barmode="group",
        title=dict(text="Segment Population Share vs. Revenue Contribution", x=0.02, xanchor="left"),
        template="plotly_dark",
        margin=dict(l=55, r=25, t=60, b=55),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
            title_text="",
        ),
        xaxis=dict(tickangle=-15),
        yaxis=dict(title="Percentage (%)", range=[0, max_pct * 1.18]),
    )
    return fig


def plot_roc_curves(roc_data: Dict[str, Dict[str, List[float]]]) -> go.Figure:
    """Plot ROC curves for Logistic Regression and Random Forest."""
    fig = go.Figure()
    # Baseline
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        line=dict(dash="dash", color="#94A3B8"),
        name="Random Guess (AUC = 0.50)",
    ))
    colors = {"logistic_regression": PALETTE["secondary"], "random_forest": PALETTE["success"]}
    for name, data in roc_data.items():
        label = "Logistic Regression" if name == "logistic_regression" else "Random Forest"
        fig.add_trace(go.Scatter(
            x=data["fpr"],
            y=data["tpr"],
            mode="lines",
            name=label,
            line=dict(color=colors.get(name, PALETTE["primary"]), width=3),
        ))
    fig.update_layout(
        title=dict(text="Receiver Operating Characteristic (ROC) Curves", x=0.02, xanchor="left"),
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        template="plotly_dark",
        margin=dict(l=65, r=25, t=55, b=60),
        legend=dict(
            x=0.52,
            y=0.08,
            xanchor="left",
            yanchor="bottom",
            bgcolor="rgba(15, 23, 42, 0.85)",
            bordercolor="#334155",
            borderwidth=1,
            font=dict(size=11),
            title_text="",
        ),
        xaxis=dict(range=[-0.02, 1.02]),
        yaxis=dict(range=[-0.02, 1.05]),
    )
    return fig


def plot_confusion_matrix_heatmap(cm: List[List[int]]) -> go.Figure:
    """Interactive Confusion Matrix heatmap with adaptive high-contrast labels."""
    labels = ["Active / Retained (0)", "Churned (1)"]
    z = np.array(cm)
    max_val = z.max() if z.size > 0 else 1
    annotations = []
    for i in range(2):
        for j in range(2):
            val = z[i, j]
            # High-contrast text: light cells get dark slate, deep blue cells get crisp white
            text_color = "#FFFFFF" if val > max_val * 0.55 else "#0F172A"
            annotations.append(
                dict(
                    x=labels[j],
                    y=labels[i],
                    text=f"<b>{val:,}</b>",
                    showarrow=False,
                    font=dict(color=text_color, size=16),
                )
            )

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=labels,
        y=labels,
        colorscale="Blues",
        showscale=False,
    ))
    fig.update_layout(
        title=dict(text="Model Confusion Matrix (Holdout Test Set)", x=0.02, xanchor="left"),
        xaxis_title="Predicted Label",
        yaxis_title="True Label",
        annotations=annotations,
        template="plotly_dark",
        margin=dict(l=85, r=25, t=55, b=55),
    )
    return fig


def plot_feature_importance(importance_dict: Dict[str, float], top_n: int = 10) -> go.Figure:
    """Horizontal bar chart for Random Forest feature importances with unclipped labels."""
    items = list(importance_dict.items())[:top_n]
    features = [k for k, _ in reversed(items)]
    scores = [v for _, v in reversed(items)]
    max_score = max(scores) if scores else 0.3

    fig = go.Figure(go.Bar(
        x=scores,
        y=features,
        orientation="h",
        marker_color=PALETTE["secondary"],
        text=[f"{s:.4f}" for s in scores],
        textposition="outside",
        cliponaxis=False,
    ))
    fig.update_layout(
        title=dict(text=f"Top {top_n} Churn Predictors (Feature Importance)", x=0.02, xanchor="left"),
        xaxis_title="Importance Score",
        template="plotly_dark",
        height=420,
        margin=dict(l=125, r=60, t=50, b=45),
        xaxis=dict(range=[0, max_score * 1.25]),
    )
    return fig


def plot_correlation_heatmap(corr_df: pd.DataFrame) -> go.Figure:
    """Correlation matrix heatmap."""
    fig = px.imshow(
        corr_df,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        title="Customer Features Correlation Heatmap",
    )
    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=90, r=25, t=50, b=65),
        title=dict(x=0.02, xanchor="left"),
    )
    return fig
