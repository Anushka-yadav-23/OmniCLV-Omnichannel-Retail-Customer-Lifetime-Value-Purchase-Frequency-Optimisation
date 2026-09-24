"""
AnushkaYadav_OmniCLV_Dashboard.py
===================================
OmniCLV – Omnichannel Customer Lifetime Value Optimisation
Interactive Streamlit Dashboard
Author : Anushka Yadav

Launch: streamlit run AnushkaYadav_OmniCLV_Dashboard.py
Prereq : Run AnushkaYadav_OmniCLV.py at least once to generate outputs/rfm_segments.csv
"""

import os
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="OmniCLV Dashboard",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS  – clean, professional look
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Page background */
    .stApp { background-color: #f7f9fc; }

    /* Metric cards */
    [data-testid="metric-container"] {
        background-color: #ffffff;
        border: 1px solid #e0e4eb;
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.07);
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #1f3a5f;
        color: white;
    }
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stMultiSelect label {
        color: #d0dff0 !important;
    }

    /* Tabs */
    .stTabs [role="tab"] { font-size: 15px; font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
DATA_FILE    = "Online Retail Data Set.csv"
XLSX_FILE    = "Online Retail Data Set.xlsx"
RFM_FILE     = os.path.join("outputs", "rfm_segments.csv")
RANDOM_SEED  = 42
CHANNELS     = ["Website", "Mobile App", "In-Store"]
SEGMENT_COLORS = {
    "Champions":       "#2ecc71",
    "Loyal Customers": "#3498db",
    "At-Risk":         "#e67e22",
    "Lost / Inactive": "#e74c3c",
}


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading transaction data …")
def load_transactions() -> pd.DataFrame:
    """Load and lightly clean the Online Retail dataset."""
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(
            DATA_FILE,
            encoding="ISO-8859-1",
            dtype={"CustomerID": str, "InvoiceNo": str},
            parse_dates=["InvoiceDate"],
        )
    elif os.path.exists(XLSX_FILE):
        df = pd.read_excel(
            XLSX_FILE,
            dtype={"CustomerID": str, "InvoiceNo": str},
            parse_dates=["InvoiceDate"],
            engine="openpyxl",
        )
    else:
        st.error(
            f"Dataset not found. Place '{DATA_FILE}' or '{XLSX_FILE}' "
            "in the project directory and restart."
        )
        st.stop()

    # Clean
    df = df.dropna(subset=["CustomerID"])
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
    df["CustomerID"] = df["CustomerID"].astype(str).str.strip().str.split(".").str[0]
    df["TotalSales"]  = df["Quantity"] * df["UnitPrice"]
    df["YearMonth"]   = df["InvoiceDate"].dt.to_period("M").astype(str)

    # Simulated channel assignment (deterministic)
    np.random.seed(RANDOM_SEED)
    channel_map = {
        cid: np.random.choice(CHANNELS, p=[0.50, 0.30, 0.20])
        for cid in df["CustomerID"].unique()
    }
    df["SalesChannel"] = df["CustomerID"].map(channel_map)

    # Derive a broad product category from description (first two words)
    df["Description"] = df["Description"].fillna("Unknown").str.strip()
    df["Category"] = df["Description"].str.split().str[:2].str.join(" ").str.upper()

    return df


@st.cache_data(show_spinner="Loading RFM segments …")
def load_rfm() -> pd.DataFrame | None:
    """Load pre-computed RFM segment file if it exists."""
    if os.path.exists(RFM_FILE):
        rfm = pd.read_csv(RFM_FILE, dtype={"CustomerID": str})
        return rfm
    return None


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR  –  Filters
# ─────────────────────────────────────────────────────────────────────────────
def render_sidebar(df: pd.DataFrame):
    """Render sidebar controls and return filter values."""
    with st.sidebar:
        st.image(
            "https://upload.wikimedia.org/wikipedia/commons/5/51/IBM_logo.svg",
            width=80,
        )
        st.markdown("## 🛍️ OmniCLV Dashboard")
        st.markdown("**Omnichannel Customer Insights**")
        st.divider()

        # Channel filter
        st.markdown("### Sales Channel")
        selected_channels = st.multiselect(
            label="Select channel(s)",
            options=CHANNELS,
            default=CHANNELS,
            label_visibility="collapsed",
        )

        # Date range filter
        st.markdown("### Date Range")
        min_date = df["InvoiceDate"].min().date()
        max_date = df["InvoiceDate"].max().date()
        start_date = st.date_input("Start date", value=min_date,
                                   min_value=min_date, max_value=max_date)
        end_date   = st.date_input("End date",   value=max_date,
                                   min_value=min_date, max_value=max_date)

        # Category filter
        st.markdown("### Product Category")
        top_cats = df["Category"].value_counts().head(30).index.tolist()
        selected_cats = st.multiselect(
            label="Select category(s)",
            options=top_cats,
            default=[],
            placeholder="All categories",
            label_visibility="collapsed",
        )

        st.divider()
        st.caption("Author: Anushka Yadav")

    return selected_channels, start_date, end_date, selected_cats


# ─────────────────────────────────────────────────────────────────────────────
# FILTER DATA
# ─────────────────────────────────────────────────────────────────────────────
def apply_filters(df, channels, start_date, end_date, categories):
    """Return filtered DataFrame based on sidebar selections."""
    mask = (
        df["SalesChannel"].isin(channels)
        & (df["InvoiceDate"].dt.date >= start_date)
        & (df["InvoiceDate"].dt.date <= end_date)
    )
    filtered = df[mask]
    if categories:
        filtered = filtered[filtered["Category"].isin(categories)]
    return filtered


# ─────────────────────────────────────────────────────────────────────────────
# KPI CARDS
# ─────────────────────────────────────────────────────────────────────────────
def render_kpis(filtered: pd.DataFrame, full: pd.DataFrame) -> None:
    """Display three KPI metric cards."""
    total_rev    = filtered["TotalSales"].sum()
    total_txn    = filtered["InvoiceNo"].nunique()
    unique_custs = filtered["CustomerID"].nunique()

    # Compare with full dataset for delta
    full_rev = full["TotalSales"].sum()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="💰 Total Revenue",
            value=f"£{total_rev:,.0f}",
            delta=f"{(total_rev / full_rev - 1):.1%} vs full dataset"
            if full_rev > 0 else None,
        )

    with col2:
        st.metric(
            label="🧾 Total Transactions",
            value=f"{total_txn:,}",
        )

    with col3:
        st.metric(
            label="👥 Unique Customers",
            value=f"{unique_custs:,}",
        )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 – REVENUE TRENDS
# ─────────────────────────────────────────────────────────────────────────────
def render_revenue_tab(filtered: pd.DataFrame) -> None:
    """Monthly Revenue Trend and Channel Performance charts."""

    st.subheader("📈 Monthly Revenue Trend")

    # Aggregate by month and channel
    monthly = (
        filtered.groupby(["YearMonth", "SalesChannel"])["TotalSales"]
        .sum()
        .reset_index()
        .sort_values("YearMonth")
    )

    if monthly.empty:
        st.info("No data for the selected filters.")
        return

    fig_trend = px.line(
        monthly,
        x="YearMonth",
        y="TotalSales",
        color="SalesChannel",
        markers=True,
        labels={"TotalSales": "Revenue (£)", "YearMonth": "Month", "SalesChannel": "Channel"},
        color_discrete_map={
            "Website":    "#4C72B0",
            "Mobile App": "#DD8452",
            "In-Store":   "#55A868",
        },
    )
    fig_trend.update_layout(
        plot_bgcolor="#ffffff",
        paper_bgcolor="#f7f9fc",
        legend_title_text="Channel",
        xaxis_tickangle=-45,
        hovermode="x unified",
        height=380,
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    st.divider()
    st.subheader("📊 Channel Performance Comparison")

    # Channel-level summary
    channel_summary = (
        filtered.groupby("SalesChannel")
        .agg(
            Revenue=("TotalSales", "sum"),
            Transactions=("InvoiceNo", "nunique"),
            AvgOrderValue=("TotalSales", "mean"),
        )
        .reset_index()
    )

    fig_ch = make_subplots(
        rows=1, cols=3,
        subplot_titles=["Total Revenue (£)", "Transactions", "Avg Order Value (£)"],
    )
    colors_map = {"Website": "#4C72B0", "Mobile App": "#DD8452", "In-Store": "#55A868"}

    for idx, metric in enumerate(["Revenue", "Transactions", "AvgOrderValue"], start=1):
        fig_ch.add_trace(
            go.Bar(
                x=channel_summary["SalesChannel"],
                y=channel_summary[metric],
                marker_color=[colors_map.get(c, "#aaa") for c in channel_summary["SalesChannel"]],
                showlegend=False,
                text=channel_summary[metric].round(0).astype(int),
                textposition="outside",
            ),
            row=1, col=idx,
        )

    fig_ch.update_layout(
        height=380,
        plot_bgcolor="#ffffff",
        paper_bgcolor="#f7f9fc",
    )
    st.plotly_chart(fig_ch, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 – CUSTOMER SEGMENTS
# ─────────────────────────────────────────────────────────────────────────────
def render_segments_tab(filtered: pd.DataFrame, rfm: pd.DataFrame | None) -> None:
    """RFM segment distribution, scatter, and churn risk histogram."""

    if rfm is None:
        st.warning(
            "RFM segment file not found (`outputs/rfm_segments.csv`). "
            "Run `python AnushkaYadav_OmniCLV.py` first to generate it.",
            icon="⚠️",
        )
        return

    # Merge RFM with channel information from filtered transactions
    rfm_cids = set(filtered["CustomerID"].unique())
    rfm_filtered = rfm[rfm["CustomerID"].isin(rfm_cids)].copy()

    if rfm_filtered.empty:
        st.info("No RFM data for the selected filters.")
        return

    col_left, col_right = st.columns([1, 2])

    # ── Pie Chart: segment distribution ────────────────────────────────────
    with col_left:
        st.subheader("Customer Segments")
        seg_counts = rfm_filtered["SegmentLabel"].value_counts().reset_index()
        seg_counts.columns = ["Segment", "Count"]

        fig_pie = px.pie(
            seg_counts,
            names="Segment",
            values="Count",
            color="Segment",
            color_discrete_map=SEGMENT_COLORS,
            hole=0.45,
        )
        fig_pie.update_traces(textinfo="percent+label", pull=[0.05] * len(seg_counts))
        fig_pie.update_layout(
            showlegend=True,
            legend_title_text="Segment",
            paper_bgcolor="#f7f9fc",
            height=370,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # ── Scatter Plot: Recency vs Monetary ───────────────────────────────────
    with col_right:
        st.subheader("Recency vs Monetary Value (log)")
        rfm_filtered["LogMonetary"] = np.log1p(rfm_filtered["Monetary"])

        fig_sc = px.scatter(
            rfm_filtered,
            x="Recency",
            y="LogMonetary",
            color="SegmentLabel",
            hover_data={
                "CustomerID": True,
                "Frequency": True,
                "Monetary": ":.2f",
                "PrimaryChannel": True,
            },
            color_discrete_map=SEGMENT_COLORS,
            labels={
                "Recency": "Recency (days since last purchase)",
                "LogMonetary": "log(Total Spend £)",
                "SegmentLabel": "Segment",
            },
            opacity=0.65,
        )
        fig_sc.update_traces(marker_size=6)
        fig_sc.update_layout(
            plot_bgcolor="#ffffff",
            paper_bgcolor="#f7f9fc",
            legend_title_text="Segment",
            height=370,
        )
        st.plotly_chart(fig_sc, use_container_width=True)

    st.divider()
    st.subheader("📉 Churn Risk Distribution by Channel")

    # Churn risk: proxy = normalised Recency score (0–1, higher = higher risk)
    rfm_filtered["ChurnRisk"] = (
        rfm_filtered["Recency"] / rfm_filtered["Recency"].max()
    ).clip(0, 1)

    fig_hist = px.histogram(
        rfm_filtered,
        x="ChurnRisk",
        color="PrimaryChannel",
        nbins=30,
        barmode="overlay",
        opacity=0.75,
        labels={"ChurnRisk": "Churn Risk Score (0 = active, 1 = high risk)", "PrimaryChannel": "Channel"},
        color_discrete_map={
            "Website":    "#4C72B0",
            "Mobile App": "#DD8452",
            "In-Store":   "#55A868",
        },
    )
    fig_hist.update_layout(
        plot_bgcolor="#ffffff",
        paper_bgcolor="#f7f9fc",
        height=350,
        xaxis_title="Churn Risk Score",
        yaxis_title="Number of Customers",
        legend_title_text="Channel",
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # ── Segment Summary Table ───────────────────────────────────────────────
    st.divider()
    st.subheader("📋 Segment Profile Summary")
    profile = (
        rfm_filtered.groupby("SegmentLabel")[["Recency", "Frequency", "Monetary"]]
        .mean()
        .round(2)
        .reset_index()
        .rename(columns={
            "SegmentLabel": "Segment",
            "Recency":      "Avg Recency (days)",
            "Frequency":    "Avg Frequency",
            "Monetary":     "Avg Monetary (£)",
        })
    )
    st.dataframe(
        profile.style.background_gradient(subset=["Avg Monetary (£)"], cmap="Blues"),
        use_container_width=True,
        hide_index=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN LAYOUT
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # ── Header ───────────────────────────────────────────────────────────────
    st.markdown(
        """
        <h1 style='color:#1f3a5f; margin-bottom:4px;'>
            🛍️ OmniCLV — Customer Lifetime Value Dashboard
        </h1>
        <p style='color:#57606a; font-size:15px; margin-top:0;'>
            Omnichannel Retail Analytics &nbsp;|&nbsp; Author: <strong>Anushka Yadav</strong>
        </p>
        <hr style='border:1px solid #e0e4eb; margin-bottom:20px;'>
        """,
        unsafe_allow_html=True,
    )

    # ── Load data ─────────────────────────────────────────────────────────────
    df  = load_transactions()
    rfm = load_rfm()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    channels, start_date, end_date, categories = render_sidebar(df)

    # ── Apply filters ─────────────────────────────────────────────────────────
    filtered = apply_filters(df, channels, start_date, end_date, categories)

    if filtered.empty:
        st.warning("No data matches the selected filters. Adjust the sidebar controls.")
        st.stop()

    # ── KPI Cards ─────────────────────────────────────────────────────────────
    render_kpis(filtered, df)
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2 = st.tabs(["📈 Revenue Trends", "👥 Customer Segments"])

    with tab1:
        render_revenue_tab(filtered)

    with tab2:
        render_segments_tab(filtered, rfm)

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown(
        """
        <hr style='border:1px solid #e0e4eb; margin-top:40px;'>
        <p style='text-align:center; color:#57606a; font-size:12px;'>
            OmniCLV Dashboard &nbsp;·&nbsp; Author: Anushka Yadav
            &nbsp;·&nbsp; Data: UCI Online Retail Dataset
        </p>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
