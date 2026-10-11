"""
InsightEngine — E-Commerce Customer Analytics Dashboard
Main Streamlit Application — Optimized High-Performance Edition
"""

import sys
from pathlib import Path
import json
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Ensure project root in sys.path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.data.loader import load_cleaned_data
from src.models.segmentation import load_segmentation_artifacts
from src.models.churn import load_churn_artifacts, FEATURE_COLUMNS
from src.models.recommender import load_recommender_artifacts
from src.analytics.eda import (
    get_summary_kpis,
    get_time_series_aggregations,
    get_geographic_aggregations,
    get_top_products_summary,
    get_bivariate_correlations,
)
from src.analytics.statistics import simulate_marketing_ab_test
from src.analytics.business_insights import generate_executive_insights
from src.analytics.precompute import load_precomputed_aggregates
from src.visualization.charts import (
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

# Page Configuration
st.set_page_config(
    page_title="InsightEngine | Customer Analytics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling — Enhanced responsive layout and typography
st.markdown("""
<style>
    .kpi-card {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 14px 16px;
        margin-bottom: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
        min-height: 110px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .kpi-title {
        color: #94A3B8;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .kpi-value {
        color: #F8FAFC;
        font-size: clamp(1.2rem, 1.55vw, 1.6rem);
        font-weight: 700;
        line-height: 1.2;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .kpi-delta {
        font-size: 0.78rem;
        color: #10B981;
        margin-top: 4px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .badge-champ { background-color: #065F46; color: #A7F3D0; padding: 3px 8px; border-radius: 6px; font-weight: 600; font-size: 0.8rem; }
    .badge-loyal { background-color: #1E40AF; color: #BFDBFE; padding: 3px 8px; border-radius: 6px; font-weight: 600; font-size: 0.8rem; }
    .badge-pot { background-color: #92400E; color: #FDE68A; padding: 3px 8px; border-radius: 6px; font-weight: 600; font-size: 0.8rem; }
    .badge-risk { background-color: #991B1B; color: #FECACA; padding: 3px 8px; border-radius: 6px; font-weight: 600; font-size: 0.8rem; }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# HIGH-PERFORMANCE CACHING LOADERS
# ==============================================================================

@st.cache_data(show_spinner=False)
def get_cached_precomputed_aggregates():
    """Load lightweight precomputed analytical aggregates (< 80 KB)."""
    return load_precomputed_aggregates()


@st.cache_data(show_spinner="Loading transactions...")
def get_cached_data():
    """Load full transaction DataFrame, optimizing datetime derivation."""
    df = load_cleaned_data()
    if "Date" not in df.columns:
        if "InvoiceDateOnly" in df.columns:
            df["Date"] = pd.to_datetime(df["InvoiceDateOnly"]).dt.date
        else:
            df["Date"] = df["InvoiceDate"].dt.date
    if "YearMonth" not in df.columns:
        df["YearMonth"] = df["InvoiceDate"].dt.strftime("%Y-%m")
    if "DayOfWeek" not in df.columns:
        df["DayOfWeek"] = df["InvoiceDate"].dt.day_name()
    if "Hour" not in df.columns:
        df["Hour"] = df["InvoiceDate"].dt.hour
    return df


@st.cache_resource(show_spinner="Loading segmentation artifacts...")
def get_cached_segmentation():
    """Load KMeans model, scaler, profiles, and customer segments."""
    return load_segmentation_artifacts()


@st.cache_resource(show_spinner="Loading churn intelligence artifacts...")
def get_cached_churn():
    """Load churn classifiers, metadata, and pre-indexed customer risk scorecard."""
    art = load_churn_artifacts()
    art["customer_churn_indexed"] = art["customer_churn_scored"].set_index("CustomerID")
    return art


@st.cache_resource(show_spinner="Loading product recommender...")
def get_cached_recommender():
    """Load ultra-compact collaborative filtering model (1.95 MB)."""
    return load_recommender_artifacts()


@st.cache_resource(show_spinner=False)
def get_cached_models():
    """Aggregate loader preserving backward compatibility."""
    return get_cached_segmentation(), get_cached_churn(), get_cached_recommender()


# Reusable cached analytical aggregations (using leading _ to avoid hashing full DataFrames)
@st.cache_data
def get_cached_summary_kpis(_df):
    return get_summary_kpis(_df)


@st.cache_data
def get_cached_time_series(_df):
    return get_time_series_aggregations(_df)


@st.cache_data
def get_cached_geographic(_df):
    return get_geographic_aggregations(_df)


@st.cache_data
def get_cached_top_products_summary(_df, top_n: int = 10):
    return get_top_products_summary(_df, top_n=top_n)


@st.cache_data
def get_cached_executive_insights(_df, _seg_df, _churn_df):
    return generate_executive_insights(_df, segmented_df=_seg_df, churn_df=_churn_df)


@st.cache_data
def get_cached_order_values(_df):
    return _df.groupby("InvoiceNo")["Revenue"].sum().values


@st.cache_data
def get_cached_correlations(_features_df):
    return get_bivariate_correlations(_features_df)


@st.cache_data
def get_filtered_eda_data(_df, selected_country: str, start_date_str: str, end_date_str: str, selected_segment: str, _seg_df: pd.DataFrame):
    filtered = _df
    if selected_country != "All Countries":
        filtered = filtered[filtered["Country"] == selected_country]
    if start_date_str and end_date_str:
        start_d = pd.to_datetime(start_date_str)
        end_d = pd.to_datetime(end_date_str) + pd.Timedelta(days=1)
        filtered = filtered[(filtered["InvoiceDate"] >= start_d) & (filtered["InvoiceDate"] <= end_d)]
    if selected_segment != "All Segments":
        seg_cids = _seg_df[_seg_df["SegmentName"] == selected_segment]["CustomerID"]
        filtered = filtered[filtered["CustomerID"].isin(seg_cids)]
    return filtered


@st.cache_data(max_entries=5)
def convert_to_csv(_df: pd.DataFrame, max_rows: int = 5000) -> bytes:
    return _df.head(max_rows).to_csv(index=False).encode("utf-8")


@st.cache_data(max_entries=10)
def convert_segment_csv(_cust_segs: pd.DataFrame, selected_seg_name: str) -> bytes:
    sub = _cust_segs[_cust_segs["SegmentName"] == selected_seg_name][["CustomerID", "Recency", "Frequency", "Monetary", "AOV", "RFM_Segment"]]
    return sub.to_csv(index=False).encode("utf-8")


@st.cache_data
def convert_churn_csv(_scored_cust: pd.DataFrame) -> bytes:
    return _scored_cust.to_csv(index=False).encode("utf-8")


@st.cache_data
def get_cached_recs(_model, customer_id, n: int = 5):
    return _model.recommend(customer_id, n=n)


@st.cache_data
def get_cached_ab_test(_order_vals, lift_input: float, random_state: int = 42):
    return simulate_marketing_ab_test(treatment_lift_pct=lift_input, random_state=random_state, order_values=_order_vals)


# Cached Chart Renderers to avoid rebuilding Plotly figures on every rerun
@st.cache_data
def get_cached_revenue_trend_fig(_monthly_df):
    return plot_revenue_trend(_monthly_df, x_col="YearMonth", y_col="Revenue")


@st.cache_data
def get_cached_geographic_fig(_geo_df):
    return plot_geographic_bar(_geo_df, top_n=8)


@st.cache_data
def get_cached_segment_comparison_fig(_prof_df):
    return plot_segment_comparison(_prof_df)


@st.cache_data
def get_cached_rfm_scatter_fig(_cust_segs):
    return plot_rfm_scatter(_cust_segs)


@st.cache_data
def get_cached_roc_fig(_roc_curves):
    return plot_roc_curves(_roc_curves)


@st.cache_data
def get_cached_confusion_matrix_fig(_cm):
    return plot_confusion_matrix_heatmap(_cm)


@st.cache_data
def get_cached_feature_importance_fig(_feat_imp):
    return plot_feature_importance(_feat_imp)


# ==============================================================================
# FAST METADATA & SIDEBAR SETUP
# ==============================================================================

# Instant load of precomputed analytical aggregates
precomputed = get_cached_precomputed_aggregates()

# Sidebar Navigation
st.sidebar.image("https://img.icons8.com/fluency/96/combo-chart.png", width=64)
st.sidebar.title("InsightEngine")
st.sidebar.caption("Enterprise E-Commerce Analytics")

nav_page = st.sidebar.radio(
    "Navigation",
    [
        "1. Executive Overview",
        "2. EDA Explorer",
        "3. Customer Segmentation",
        "4. Churn Intelligence",
        "5. Product Recommendations",
        "6. Statistical Analysis",
        "7. Business Insights",
        "8. Methodology & Architecture",
    ],
)

st.sidebar.markdown("---")
st.sidebar.caption(f"📊 **Database**: UCI Online Retail")
st.sidebar.caption(f"📦 **Transactions**: {precomputed['total_transactions']:,}")
st.sidebar.caption(f"👥 **Customers**: {precomputed['total_customers']:,}")


# ==============================================================================
# PAGE 1: EXECUTIVE OVERVIEW
# ==============================================================================
if nav_page.startswith("1."):
    st.title("⚡ Executive Overview")
    st.markdown("High-level performance monitoring, customer portfolio health, and executive summary.")

    kpis = precomputed["kpis"]
    churn_artifacts = get_cached_churn()
    churn_rate = churn_artifacts["metadata"].get("dataset_churn_rate", 0.43)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Total Revenue</div>
            <div class="kpi-value">${kpis['total_revenue']:,.0f}</div>
            <div class="kpi-delta">▲ 100% audited sales</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Active Customers</div>
            <div class="kpi-value">{kpis['total_customers']:,}</div>
            <div class="kpi-delta">▲ Distinct customer accounts</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Total Orders</div>
            <div class="kpi-value">{kpis['total_orders']:,}</div>
            <div class="kpi-delta">▲ Valid invoices</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Catalog SKUs</div>
            <div class="kpi-value">{kpis['total_products']:,}</div>
            <div class="kpi-delta">Unique products</div>
        </div>
        """, unsafe_allow_html=True)
    with c5:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Average Order (AOV)</div>
            <div class="kpi-value">${kpis['avg_order_value']:.2f}</div>
            <div class="kpi-delta">Basket spend per order</div>
        </div>
        """, unsafe_allow_html=True)
    with c6:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Churn Rate</div>
            <div class="kpi-value">{churn_rate:.1%}</div>
            <div class="kpi-delta" style="color: #EF4444;">▼ Inactivity forward window</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 📈 Revenue Trend & Geographic Footprint")
    ts = precomputed["time_series"]
    col_rev, col_geo = st.columns([1.5, 1])
    with col_rev:
        st.plotly_chart(get_cached_revenue_trend_fig(ts["monthly"]), use_container_width=True)
    with col_geo:
        geo = precomputed["geographic"]
        st.plotly_chart(get_cached_geographic_fig(geo), use_container_width=True)

    st.markdown("### 👥 Customer Portfolio & Segment Contribution")
    col_seg, col_ins = st.columns([1.3, 1])
    with col_seg:
        seg_artifacts = get_cached_segmentation()
        prof_df = seg_artifacts["profiles"]
        st.plotly_chart(get_cached_segment_comparison_fig(prof_df), use_container_width=True)
    with col_ins:
        st.markdown("#### 💡 Key Executive Insights")
        exec_ins = precomputed["executive_insights"]
        st.info(f"**Pareto Concentration**: {exec_ins['kpis']['pareto_share_pct']}% of revenue is driven by the top 20% of accounts.")
        st.warning(f"**Geographic Risk**: {exec_ins['kpis']['top_country']} drives {exec_ins['kpis']['top_country_pct']}% of total company sales.")
        st.success(f"**Top Segment Power**: Champions account for only {prof_df.loc[prof_df['SegmentName']=='Champions', 'CustomerPct'].values[0]}% of customers, but yield {prof_df.loc[prof_df['SegmentName']=='Champions', 'RevenuePct'].values[0]}% of total revenue.")


# ==============================================================================
# PAGE 2: EDA EXPLORER
# ==============================================================================
elif nav_page.startswith("2."):
    st.title("🔍 Exploratory Data Analysis (EDA) Explorer")
    st.markdown("Interactive slice-and-dice of sales, orders, quantities, geography, and correlations.")

    seg_artifacts = get_cached_segmentation()
    churn_artifacts = get_cached_churn()

    # Sidebar / Top filters
    with st.expander("⚙️ Interactive Filters", expanded=True):
        f1, f2, f3 = st.columns(3)
        with f1:
            all_countries = ["All Countries"] + precomputed["all_countries"]
            selected_country = st.selectbox("Country Filter", all_countries)
        with f2:
            min_date = precomputed["min_date"]
            max_date = precomputed["max_date"]
            date_range = st.date_input("Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)
        with f3:
            cust_segments = ["All Segments"] + sorted(seg_artifacts["customer_segments"]["SegmentName"].unique().tolist())
            selected_segment = st.selectbox("Customer Segment", cust_segments)

    # Filter application using cached helper
    start_date_str = str(date_range[0]) if len(date_range) >= 1 else ""
    end_date_str = str(date_range[1]) if len(date_range) == 2 else start_date_str

    is_default_filter = (
        selected_country == "All Countries"
        and len(date_range) == 2
        and date_range[0] == min_date
        and date_range[1] == max_date
        and selected_segment == "All Segments"
    )

    df_clean = get_cached_data()

    if is_default_filter:
        df_filtered = df_clean
        total_revenue_filtered = precomputed["total_revenue"]
        ts_filt = precomputed["time_series"]
        top_prods = precomputed["top_products"]
    else:
        df_filtered = get_filtered_eda_data(
            df_clean,
            selected_country,
            start_date_str,
            end_date_str,
            selected_segment,
            seg_artifacts["customer_segments"],
        )
        total_revenue_filtered = df_filtered['Revenue'].sum()
        ts_filt = get_cached_time_series(df_filtered)
        top_prods = get_cached_top_products_summary(df_filtered, top_n=10)

    st.write(f"Showing **{len(df_filtered):,}** matching transactions (${total_revenue_filtered:,.2f} total revenue).")

    tab1, tab2, tab3, tab4 = st.tabs(["Distributions & Boxplots", "Time-Series Patterns", "Top Products", "Feature Correlations"])

    with tab1:
        c_hist1, c_hist2 = st.columns(2)
        with c_hist1:
            rev_subset = df_filtered[df_filtered["Revenue"] < 300]
            fig_rev = px.histogram(
                rev_subset,
                x="Revenue",
                nbins=50,
                title="Revenue Distribution (Orders < $300)",
                template="plotly_dark",
                color_discrete_sequence=[PALETTE["primary"]],
            )
            fig_rev.update_layout(
                margin=dict(l=40, r=20, t=50, b=30),
                title=dict(x=0.02, xanchor="left"),
            )
            st.plotly_chart(fig_rev, use_container_width=True)
        with c_hist2:
            qty_subset = df_filtered[df_filtered["Quantity"] < 100]
            # Downsample for boxplot if over 5,000 points to keep WebSocket payload lightweight
            qty_box_data = qty_subset.sample(5000, random_state=42) if len(qty_subset) > 5000 else qty_subset
            fig_qty = px.box(
                qty_box_data,
                y="Quantity",
                title="Quantity per Line Item (Boxplot)",
                template="plotly_dark",
                color_discrete_sequence=[PALETTE["secondary"]],
            )
            fig_qty.update_layout(
                margin=dict(l=40, r=20, t=50, b=30),
                title=dict(x=0.02, xanchor="left"),
            )
            st.plotly_chart(fig_qty, use_container_width=True)

    with tab2:
        c_ts1, c_ts2 = st.columns(2)
        with c_ts1:
            fig_dow = px.bar(
                ts_filt["day_of_week"],
                x="DayOfWeek",
                y="Revenue",
                title="Revenue by Day of Week",
                template="plotly_dark",
                color="Revenue",
                color_continuous_scale="Viridis",
            )
            fig_dow.update_layout(
                margin=dict(l=40, r=20, t=50, b=30),
                title=dict(x=0.02, xanchor="left"),
            )
            st.plotly_chart(fig_dow, use_container_width=True)
        with c_ts2:
            fig_hr = px.line(
                ts_filt["hourly"],
                x="Hour",
                y="Orders",
                markers=True,
                title="Order Volume by Hour of Day",
                template="plotly_dark",
                color_discrete_sequence=[PALETTE["accent"]],
            )
            fig_hr.update_layout(
                margin=dict(l=40, r=20, t=50, b=30),
                title=dict(x=0.02, xanchor="left"),
            )
            st.plotly_chart(fig_hr, use_container_width=True)

    with tab3:
        c_tp1, c_tp2 = st.columns(2)
        with c_tp1:
            st.markdown("#### Top Products by Revenue")
            st.dataframe(top_prods["by_revenue"][["StockCode", "Description", "TotalRevenue", "TotalQuantity"]], use_container_width=True)
        with c_tp2:
            st.markdown("#### Top Products by Quantity")
            st.dataframe(top_prods["by_quantity"][["StockCode", "Description", "TotalQuantity", "TotalRevenue"]], use_container_width=True)

    with tab4:
        cust_feats = churn_artifacts["customer_churn_scored"]
        corr_matrix = get_cached_correlations(cust_feats)
        st.plotly_chart(plot_correlation_heatmap(corr_matrix), use_container_width=True)

    # Cached CSV Download
    csv_data = convert_to_csv(df_filtered, max_rows=5000)
    st.download_button("📥 Download Filtered Transactions (CSV)", data=csv_data, file_name="filtered_transactions.csv", mime="text/csv")


# ==============================================================================
# PAGE 3: CUSTOMER SEGMENTATION
# ==============================================================================
elif nav_page.startswith("3."):
    st.title("🎯 RFM Customer Segmentation")
    st.markdown("K-Means unsupervised clustering on normalized RFM (Recency, Frequency, Monetary) vectors.")

    seg_artifacts = get_cached_segmentation()
    prof_df = seg_artifacts["profiles"]
    cust_segs = seg_artifacts["customer_segments"]

    st.markdown("### 📊 Segment Performance Matrix")
    st.dataframe(
        prof_df[[
            "SegmentName", "CustomerCount", "CustomerPct", "TotalRevenue", "RevenuePct",
            "AvgRecency", "AvgFrequency", "AvgMonetary", "RecommendedAction"
        ]],
        use_container_width=True,
    )

    c_scat, c_bar = st.columns([1.5, 1])
    with c_scat:
        st.plotly_chart(get_cached_rfm_scatter_fig(cust_segs), use_container_width=True)
    with c_bar:
        st.plotly_chart(get_cached_segment_comparison_fig(prof_df), use_container_width=True)

    st.markdown("### 🔍 Segment Deep Dive & Action Playbook")

    @st.fragment
    def render_segment_deep_dive():
        selected_seg_name = st.selectbox("Select Segment to Inspect", prof_df["SegmentName"].tolist())
        seg_info = prof_df[prof_df["SegmentName"] == selected_seg_name].iloc[0]

        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Customer Accounts", f"{seg_info['CustomerCount']:,} ({seg_info['CustomerPct']}%)")
        s2.metric("Revenue Generated", f"${seg_info['TotalRevenue']:,.2f} ({seg_info['RevenuePct']}%)")
        s3.metric("Avg Inactivity (Recency)", f"{seg_info['AvgRecency']:.0f} days")
        s4.metric("Avg Frequency", f"{seg_info['AvgFrequency']:.1f} orders")

        st.success(f"**Recommended Action**: {seg_info['RecommendedAction']}")

        # Segment Customer Table
        seg_sample = cust_segs[cust_segs["SegmentName"] == selected_seg_name][["CustomerID", "Recency", "Frequency", "Monetary", "AOV", "RFM_Segment"]]
        st.dataframe(seg_sample.head(50), use_container_width=True)
        
        csv_seg = convert_segment_csv(cust_segs, selected_seg_name)
        st.download_button(f"📥 Download {selected_seg_name} Customers (CSV)", data=csv_seg, file_name=f"{selected_seg_name}_customers.csv", mime="text/csv")

    render_segment_deep_dive()


# ==============================================================================
# PAGE 4: CHURN INTELLIGENCE
# ==============================================================================
elif nav_page.startswith("4."):
    st.title("🚨 Churn Intelligence & ML Risk Scoring")
    st.markdown("Supervised Machine Learning (Random Forest vs Logistic Regression) with leak-free time-aware inactivity evaluation.")

    churn_artifacts = get_cached_churn()
    churn_meta = churn_artifacts["metadata"]
    scored_cust = churn_artifacts["customer_churn_scored"]
    scored_cust_indexed = churn_artifacts["customer_churn_indexed"]
    risk_counts = churn_meta["risk_counts"]
    metrics = churn_meta["metrics"]

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Evaluated Accounts", f"{churn_meta['total_customers_evaluated']:,}")
    r2.metric("High-Risk Accounts", f"{risk_counts.get('High Risk', 0):,}", delta="Urgent Intervention", delta_color="inverse")
    r3.metric("Medium-Risk Accounts", f"{risk_counts.get('Medium Risk', 0):,}")
    r4.metric("Low-Risk Accounts", f"{risk_counts.get('Low Risk', 0):,}", delta="Healthy Accounts")

    tab_eval, tab_model, tab_drill = st.tabs(["Model Performance & ROC", "Feature Importance & Coefficients", "Customer Risk Lookup"])

    with tab_eval:
        c_m1, c_m2 = st.columns(2)
        with c_m1:
            st.plotly_chart(get_cached_roc_fig(churn_meta["roc_curves"]), use_container_width=True)
        with c_m2:
            best_cm = metrics[churn_meta["best_model_name"]]["confusion_matrix"]
            st.plotly_chart(get_cached_confusion_matrix_fig(best_cm), use_container_width=True)

        st.markdown("#### 🏆 Model Benchmark Comparison")
        comp_data = {
            "Model": ["Logistic Regression", "Random Forest (Selected)"],
            "Accuracy": [metrics["logistic_regression"]["accuracy"], metrics["random_forest"]["accuracy"]],
            "Precision": [metrics["logistic_regression"]["precision"], metrics["random_forest"]["precision"]],
            "Recall": [metrics["logistic_regression"]["recall"], metrics["random_forest"]["recall"]],
            "F1-Score": [metrics["logistic_regression"]["f1"], metrics["random_forest"]["f1"]],
            "ROC-AUC": [metrics["logistic_regression"]["roc_auc"], metrics["random_forest"]["roc_auc"]],
            "Hyperparameters": [str(metrics["logistic_regression"]["best_params"]), str(metrics["random_forest"]["best_params"])],
        }
        st.dataframe(pd.DataFrame(comp_data), use_container_width=True)

    with tab_model:
        c_imp, c_coef = st.columns(2)
        with c_imp:
            st.plotly_chart(get_cached_feature_importance_fig(churn_meta["feature_importance_rf"]), use_container_width=True)
        with c_coef:
            st.markdown("#### Logistic Regression Feature Coefficients")
            coef_df = pd.DataFrame(list(churn_meta["coefficients_lr"].items()), columns=["Feature", "Log-Odds Coefficient"]).sort_values("Log-Odds Coefficient", ascending=False)
            st.dataframe(coef_df, use_container_width=True)

    with tab_drill:
        @st.fragment
        def render_churn_drilldown():
            st.markdown("#### Individual Customer Churn Scorecard")
            sample_cids = scored_cust["CustomerID"].tolist()[:50]
            selected_cid = st.selectbox("Select Customer ID", sample_cids)

            c_row = scored_cust_indexed.loc[selected_cid]
            prob = float(c_row["ChurnProbability"])
            tier = str(c_row["RiskTier"])

            col_score1, col_score2, col_score3 = st.columns(3)
            col_score1.metric("Churn Probability", f"{prob:.1%}")
            col_score2.metric("Assigned Risk Tier", tier)
            col_score3.metric("Customer Spend", f"${c_row['Monetary']:,.2f}")

            if tier == "High Risk":
                st.error(f"🚨 **High Risk Alert**: Customer {selected_cid} shows severe dormancy (Recency: {c_row['Recency']} days, Purchase Velocity: {c_row['PurchaseVelocity']:.2f}). Trigger immediate win-back discount flow.")
            elif tier == "Medium Risk":
                st.warning(f"⚠️ **Medium Risk Notice**: Customer {selected_cid} is showing signs of cooling off. Send curated recommendations.")
            else:
                st.success(f"✅ **Low Risk**: Customer {selected_cid} has healthy repeat purchase cadence.")

            st.dataframe(c_row.to_frame().T, use_container_width=True)

        render_churn_drilldown()

    csv_churn = convert_churn_csv(scored_cust)
    st.download_button("📥 Download Churn Risk Predictions (CSV)", data=csv_churn, file_name="churn_predictions.csv", mime="text/csv")


# ==============================================================================
# PAGE 5: RECOMMENDATIONS
# ==============================================================================
elif nav_page.startswith("5."):
    st.title("🛍️ Product Recommendation Engine")
    st.markdown("Item-to-Item Collaborative Filtering based on co-purchase cosine similarity matrix with cold-start bestseller fallback.")

    rec_artifacts = get_cached_recommender()
    rec_engine = rec_artifacts["model"]
    all_known_customers = rec_engine.known_customers

    @st.fragment
    def render_recs_section():
        col_input, col_meta = st.columns([1, 1])
        with col_input:
            default_cid = str(all_known_customers[0]) if all_known_customers else "17850"
            selected_cust_input = st.text_input("Enter Customer ID", value=default_cid)
            st.caption(f"Suggested Customer IDs: {', '.join([str(x) for x in all_known_customers[:6]])}")
            num_recs = st.slider("Number of Recommendations", 3, 10, 5)

        try:
            cid_val = int(selected_cust_input)
        except ValueError:
            cid_val = selected_cust_input

        recommendations = get_cached_recs(rec_engine, cid_val, num_recs)

        st.markdown(f"### 🎯 Top {num_recs} Personalized Recommendations for Customer `{selected_cust_input}`")
        
        for i, r in enumerate(recommendations, start=1):
            with st.container():
                st.markdown(f"""
                <div style="background-color: #1E293B; border-left: 4px solid #4F46E5; padding: 14px 18px; margin-bottom: 10px; border-radius: 6px;">
                    <span style="color: #06B6D4; font-weight: 700; font-size: 1.1rem;">#{i} [{r['StockCode']}] {r['Description']}</span><br/>
                    <span style="color: #94A3B8; font-size: 0.85rem;">Affinity Score: <b>{r['Score']:.4f}</b> | <i>{r['Reason']}</i></span>
                </div>
                """, unsafe_allow_html=True)

    render_recs_section()

    st.markdown("### 🔥 Store Bestsellers Baseline (Cold-Start Fallback)")
    pop_df = pd.DataFrame(rec_artifacts["popular_products"])
    st.dataframe(pop_df[["StockCode", "Description", "OrderCount", "TotalRevenue"]].head(10), use_container_width=True)


# ==============================================================================
# PAGE 6: STATISTICAL ANALYSIS
# ==============================================================================
elif nav_page.startswith("6."):
    st.title("🧪 Statistical Testing & A/B Experiment Simulation")
    st.markdown("Rigorous hypothesis testing comparing order values between control and recommendation treatment cohorts.")

    order_values_cached = precomputed["order_values"]

    @st.fragment
    def render_ab_section():
        col_sim_btn, _ = st.columns([1, 2])
        with col_sim_btn:
            lift_input = st.slider("Simulated Treatment Uplift (%)", 2.0, 15.0, 7.5, 0.5)

        ab_res = get_cached_ab_test(order_values_cached, lift_input, random_state=42)

        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Control Mean", f"${ab_res['group_a']['mean']:.2f}", f"n = {ab_res['group_a']['sample_size']:,}")
        s2.metric("Treatment Mean", f"${ab_res['group_b']['mean']:.2f}", f"+{ab_res['percentage_lift']:.1f}% Lift")
        s3.metric("Welch's t-test p-value", f"{ab_res['t_test']['p_value']:.4e}")
        s4.metric("Cohen's d (Effect Size)", f"{ab_res['effect_size_cohens_d']:.4f}")

        st.markdown("### 📋 Hypothesis & Decision Framework")
        st.info(f"**Experiment Null Hypothesis ($H_0$)**: There is no difference in mean order value between Control and Treatment ($p > 0.05$).")
        
        if ab_res["is_statistically_significant"]:
            st.success(f"**Conclusion**: {ab_res['decision_summary']}")
            st.write(f"**Recommendation**: {ab_res['business_recommendation']}")
        else:
            st.warning(f"**Conclusion**: {ab_res['decision_summary']}")
            st.write(f"**Recommendation**: {ab_res['business_recommendation']}")

        st.markdown("### 📊 Test Metrics Summary")
        metrics_tbl = [
            {"Metric / Parameter": "Metric Evaluated", "Value": ab_res["metric_name"]},
            {"Metric / Parameter": "Group A (Control) Name", "Value": ab_res["group_a"]["name"]},
            {"Metric / Parameter": "Group B (Treatment) Name", "Value": ab_res["group_b"]["name"]},
            {"Metric / Parameter": "Difference in Means", "Value": f"${ab_res['difference_in_means']:.2f}"},
            {"Metric / Parameter": "95% Confidence Interval", "Value": f"[${ab_res['confidence_interval_95'][0]}, ${ab_res['confidence_interval_95'][1]}]"},
            {"Metric / Parameter": "t-statistic", "Value": str(ab_res["t_test"]["t_statistic"])},
            {"Metric / Parameter": "t-test p-value", "Value": f"{ab_res['t_test']['p_value']:.4e}"},
            {"Metric / Parameter": "Mann-Whitney U p-value", "Value": f"{ab_res['mann_whitney_u']['p_value']:.4e}"},
            {"Metric / Parameter": "Cohen's d", "Value": str(ab_res["effect_size_cohens_d"])},
            {"Metric / Parameter": "Statistically Significant at α = 0.05", "Value": "Yes" if ab_res["is_statistically_significant"] else "No"},
        ]
        st.dataframe(pd.DataFrame(metrics_tbl), use_container_width=True)

    render_ab_section()


# ==============================================================================
# PAGE 7: BUSINESS INSIGHTS
# ==============================================================================
elif nav_page.startswith("7."):
    st.title("💼 Executive Business Insights & Action Plan")
    st.markdown("Automated data-driven strategic intelligence translated into concrete business initiatives.")

    exec_ins = precomputed["executive_insights"]
    narrative = exec_ins["narrative"]

    q1, q2 = st.columns(2)
    with q1:
        st.markdown("#### 1. What is happening?")
        st.write(narrative["what_is_happening"])
        
        st.markdown("#### 3. Which customers matter most?")
        st.write(narrative["which_customers_matter_most"])
        
        st.markdown("#### 5. What products drive performance?")
        st.write(narrative["what_products_drive_performance"])
    with q2:
        st.markdown("#### 2. Why does it matter?")
        st.write(narrative["why_it_matters"])
        
        st.markdown("#### 4. Which customers are at risk?")
        st.write(narrative["which_customers_are_at_risk"])
        
        st.markdown("#### 6. What should the business do next?")
        st.write("Deploy the prioritized action matrix below across Retention, Product, and Marketing teams.")

    st.markdown("### 🚀 Prioritized Strategic Initiatives")
    for strat in narrative["strategic_recommendations"]:
        with st.container():
            st.markdown(f"""
            <div style="background-color: #1E293B; border-left: 4px solid #10B981; padding: 14px 18px; margin-bottom: 12px; border-radius: 6px;">
                <h4 style="color: #F8FAFC; margin-bottom: 4px;">{strat['pillar']}</h4>
                <p style="color: #CBD5E1; margin-bottom: 6px;"><b>Action:</b> {strat['action']}</p>
                <span style="color: #10B981; font-weight: 600; font-size: 0.85rem;">🎯 Target Impact: {strat['target_metric']}</span>
            </div>
            """, unsafe_allow_html=True)


# ==============================================================================
# PAGE 8: METHODOLOGY & ARCHITECTURE
# ==============================================================================
elif nav_page.startswith("8."):
    st.title("📚 Methodology & System Architecture")
    st.markdown("End-to-end technical documentation, machine learning specifications, and reproducibility guarantees.")

    st.markdown("""
    ### 🏗️ Pipeline Architecture
    ```text
    Dataset (UCI Online Retail)
       │
       ▼
    Data Loading & Validation (src.data.loader)
       │
       ▼
    Data Cleaning & Audit (src.data.cleaning)
       ├── Filter Cancellations ('C' Invoices)
       ├── Filter Non-positive Quantity & Price
       ├── Remove Missing CustomerID
       └── Calculate Revenue = Quantity × UnitPrice
       │
       ▼
    Processed Dataset (data/processed/cleaned_ecommerce.csv)
       │
       ├─────────────────────────┬─────────────────────────┬─────────────────────────┐
       ▼                         ▼                         ▼                         ▼
    Exploratory Data        RFM Feature               Time-Aware Churn          Collaborative
    Analysis (src.analytics) Engineering              Engineering               Filtering Recommender
       │                         │                         │                         │
       ▼                         ▼                         ▼                         ▼
    Interactive Trends      K-Means Clustering        Random Forest & LogReg    Item-Item Cosine
    & Distributions         (Optimal K, Profiles)     (GridSearchCV, ROC-AUC)   Similarity & Cold Start
       │                         │                         │                         │
       └─────────────────────────┴─────────────────────────┴─────────────────────────┘
                                           │
                                           ▼
                                 Streamlit Dashboard (app.py)
                                           │
                                           ▼
                                  Actionable Insights
    ```
    """)

    st.markdown("""
    ### 🔬 Machine Learning Specifications
    - **Customer Segmentation**: Log-transformed RFM dimensions scaled via StandardScaler, clustered using K-Means with k-means++ initialization. Evaluated via Silhouette Score and Inertia Elbow method.
    - **Churn Modeling**: Non-contractual forward inactivity window (90 days). Strictly separates feature generation history from future target evaluation to avoid lookahead data leakage.
    - **Recommendation System**: Item-Item Collaborative Filtering using cosine similarity on log-transformed user-item interaction matrix, backed by store bestseller cold-start fallback.
    - **Reproducibility**: All models and pipelines run via `python -m src.pipeline` and validate via `python -m pytest tests`.
    """)
