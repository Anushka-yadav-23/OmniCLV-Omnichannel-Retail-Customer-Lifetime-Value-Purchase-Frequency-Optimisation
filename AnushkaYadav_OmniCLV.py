"""
AnushkaYadav_OmniCLV.py
========================
Omnichannel Retail Customer Lifetime Value & Purchase Frequency Optimization
Author : Anushka Yadav
Dataset: Online Retail Data Set.csv  (UCI / Kaggle Online Retail II)

Pipeline
--------
1. Data Loading & Cleaning
2. Feature Engineering  (TotalSales, SalesChannel simulation)
3. RFM Metric Calculation
4. Exploratory Data Analysis (EDA)
5. Customer Segmentation  – K-Means Clustering on RFM scores
6. Churn Classification   – Random Forest on RFM + channel features
7. Output artefacts: plots saved to ./outputs/
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend – safe for scripts
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 0.  CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
DATA_FILE   = "Online Retail Data Set.csv"   # CSV version expected at runtime
XLSX_FILE   = "Online Retail Data Set.xlsx"  # fallback if CSV absent
OUTPUT_DIR  = "outputs"
RANDOM_SEED = 42
N_CLUSTERS  = 4       # RFM K-Means segments

os.makedirs(OUTPUT_DIR, exist_ok=True)
sns.set_theme(style="whitegrid", palette="muted")


# ─────────────────────────────────────────────────────────────────────────────
# 1.  DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
def load_data(csv_path: str, xlsx_path: str) -> pd.DataFrame:
    """Load the Online Retail dataset from CSV or XLSX fallback."""
    if os.path.exists(csv_path):
        print(f"[INFO] Loading data from CSV: {csv_path}")
        df = pd.read_csv(
            csv_path,
            encoding="ISO-8859-1",
            dtype={"CustomerID": str, "InvoiceNo": str},
            parse_dates=["InvoiceDate"],
        )
    elif os.path.exists(xlsx_path):
        print(f"[INFO] CSV not found – loading from XLSX: {xlsx_path}")
        df = pd.read_excel(
            xlsx_path,
            dtype={"CustomerID": str, "InvoiceNo": str},
            parse_dates=["InvoiceDate"],
            engine="openpyxl",
        )
    else:
        raise FileNotFoundError(
            f"Neither '{csv_path}' nor '{xlsx_path}' found. "
            "Please place the dataset in the working directory."
        )

    print(f"[INFO] Raw shape: {df.shape}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2.  DATA CLEANING
# ─────────────────────────────────────────────────────────────────────────────
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleaning steps:
      - Drop rows with missing CustomerID (cannot map to a customer)
      - Remove return / cancellation invoices (InvoiceNo starts with 'C')
      - Remove rows with Quantity <= 0 or UnitPrice <= 0
      - Strip whitespace from string columns
    """
    initial_rows = len(df)

    # Remove rows without a CustomerID
    df = df.dropna(subset=["CustomerID"])
    print(f"[CLEAN] Dropped {initial_rows - len(df):,} rows with missing CustomerID")

    # Remove cancellations / returns (invoice prefix 'C')
    mask_returns = df["InvoiceNo"].astype(str).str.startswith("C")
    df = df[~mask_returns]
    print(f"[CLEAN] Dropped {mask_returns.sum():,} return/cancellation rows")

    # Remove non-positive quantities and prices
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]

    # Normalise CustomerID to integer-string for consistency
    df["CustomerID"] = df["CustomerID"].astype(str).str.strip().str.split(".").str[0]

    # Strip description whitespace
    if "Description" in df.columns:
        df["Description"] = df["Description"].str.strip()

    print(f"[CLEAN] Final shape after cleaning: {df.shape}")
    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    New columns:
      TotalSales   – Quantity × UnitPrice (line-level revenue)
      SalesChannel – Simulated channel label (Website / Mobile App / In-Store)
                     The UCI dataset has no channel column; we deterministically
                     assign channels using a hash of CustomerID so results are
                     reproducible and proportions are realistic (~50/30/20 split).
      YearMonth    – Period label for trend analysis
    """
    # Revenue per line
    df["TotalSales"] = df["Quantity"] * df["UnitPrice"]

    # Simulate omnichannel labels (deterministic, reproducible)
    np.random.seed(RANDOM_SEED)
    channel_map = {
        cid: np.random.choice(
            ["Website", "Mobile App", "In-Store"],
            p=[0.50, 0.30, 0.20],
        )
        for cid in df["CustomerID"].unique()
    }
    df["SalesChannel"] = df["CustomerID"].map(channel_map)

    # Month-year period for trend plots
    df["YearMonth"] = df["InvoiceDate"].dt.to_period("M")

    print("[FEAT] Feature engineering complete.")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4.  RFM CALCULATION
# ─────────────────────────────────────────────────────────────────────────────
def compute_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Recency, Frequency, and Monetary value per customer.

    Recency  – days since last purchase (lower = more recent)
    Frequency – number of unique invoices
    Monetary  – total spend
    """
    # Snapshot date = day after the last transaction in the dataset
    snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)
    print(f"[RFM] Snapshot date: {snapshot_date.date()}")

    rfm = (
        df.groupby("CustomerID")
        .agg(
            Recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
            Frequency=("InvoiceNo", "nunique"),
            Monetary=("TotalSales", "sum"),
        )
        .reset_index()
    )

    # Add dominant channel per customer
    channel_mode = (
        df.groupby("CustomerID")["SalesChannel"]
        .agg(lambda x: x.value_counts().index[0])
        .reset_index()
        .rename(columns={"SalesChannel": "PrimaryChannel"})
    )
    rfm = rfm.merge(channel_mode, on="CustomerID", how="left")

    print(f"[RFM] Computed RFM for {len(rfm):,} unique customers.")
    print(rfm.describe().round(2))
    return rfm


# ─────────────────────────────────────────────────────────────────────────────
# 5.  EXPLORATORY DATA ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
def run_eda(df: pd.DataFrame, rfm: pd.DataFrame) -> None:
    """
    Generates and saves EDA plots:
      a) Monthly Revenue Trend
      b) Revenue by Sales Channel
      c) RFM Distributions
      d) Top 10 Countries by Revenue
    """
    print("[EDA] Running exploratory data analysis …")

    # ── (a) Monthly Revenue Trend ──────────────────────────────────────────
    monthly = (
        df.groupby("YearMonth")["TotalSales"]
        .sum()
        .reset_index()
    )
    monthly["YearMonth"] = monthly["YearMonth"].astype(str)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(monthly["YearMonth"], monthly["TotalSales"] / 1_000, marker="o", linewidth=2)
    ax.set_title("Monthly Revenue Trend")
    ax.set_xlabel("Month")
    ax.set_ylabel("Revenue (£ thousands)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "eda_monthly_revenue.png"), dpi=150)
    plt.close(fig)

    # ── (b) Revenue by Sales Channel ──────────────────────────────────────
    channel_rev = df.groupby("SalesChannel")["TotalSales"].sum().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    channel_rev.plot(kind="bar", ax=ax, color=["#4C72B0", "#DD8452", "#55A868"])
    ax.set_title("Total Revenue by Sales Channel")
    ax.set_xlabel("Channel")
    ax.set_ylabel("Revenue (£)")
    ax.bar_label(ax.containers[0], fmt="£{:,.0f}", padding=3, fontsize=8)
    plt.xticks(rotation=0)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "eda_channel_revenue.png"), dpi=150)
    plt.close(fig)

    # ── (c) RFM Distributions ─────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col, color in zip(axes, ["Recency", "Frequency", "Monetary"],
                               ["#4C72B0", "#DD8452", "#55A868"]):
        data = rfm[col].clip(upper=rfm[col].quantile(0.99))  # cap outliers for viz
        ax.hist(data, bins=40, color=color, edgecolor="white")
        ax.set_title(f"{col} Distribution")
        ax.set_xlabel(col)
        ax.set_ylabel("# Customers")
    plt.suptitle("RFM Feature Distributions (99th-pct capped)", y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "eda_rfm_distributions.png"), dpi=150)
    plt.close(fig)

    # ── (d) Top 10 Countries by Revenue ───────────────────────────────────
    top_countries = (
        df.groupby("Country")["TotalSales"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=top_countries.values / 1_000, y=top_countries.index, ax=ax, palette="Blues_d")
    ax.set_title("Top 10 Countries by Revenue")
    ax.set_xlabel("Revenue (£ thousands)")
    ax.set_ylabel("Country")
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "eda_top_countries.png"), dpi=150)
    plt.close(fig)

    print(f"[EDA] Plots saved to ./{OUTPUT_DIR}/")


# ─────────────────────────────────────────────────────────────────────────────
# 6.  CUSTOMER SEGMENTATION – K-MEANS ON RFM SCORES
# ─────────────────────────────────────────────────────────────────────────────
def segment_customers_kmeans(rfm: pd.DataFrame) -> pd.DataFrame:
    """
    Segment customers into N_CLUSTERS groups using K-Means clustering on
    log-transformed, standardised RFM features.

    Returns the rfm DataFrame with an added 'Segment' column and human-readable
    'SegmentLabel' column.
    """
    print(f"\n[MODEL] Running K-Means clustering (k={N_CLUSTERS}) …")

    # Log-transform skewed monetary and frequency columns
    rfm_model = rfm[["Recency", "Frequency", "Monetary"]].copy()
    rfm_model["Frequency"] = np.log1p(rfm_model["Frequency"])
    rfm_model["Monetary"]  = np.log1p(rfm_model["Monetary"])

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(rfm_model)

    # Elbow plot – saved for reporting
    inertia = []
    k_range = range(2, 9)
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
        km.fit(X_scaled)
        inertia.append(km.inertia_)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(list(k_range), inertia, marker="o", linewidth=2)
    ax.set_title("K-Means Elbow Curve")
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Inertia")
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "kmeans_elbow.png"), dpi=150)
    plt.close(fig)

    # Fit final model
    km_final = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_SEED, n_init=10)
    rfm["Segment"] = km_final.fit_predict(X_scaled)

    # Label segments by mean Monetary (descending) for business-friendly names
    segment_means = rfm.groupby("Segment")["Monetary"].mean().sort_values(ascending=False)
    label_map = {
        seg: lbl
        for seg, lbl in zip(
            segment_means.index,
            ["Champions", "Loyal Customers", "At-Risk", "Lost / Inactive"],
        )
    }
    rfm["SegmentLabel"] = rfm["Segment"].map(label_map)

    # Segment profile summary
    profile = rfm.groupby("SegmentLabel")[["Recency", "Frequency", "Monetary"]].mean().round(2)
    print("[MODEL] Segment Profiles:\n", profile)

    # Visualise segment sizes
    fig, ax = plt.subplots(figsize=(7, 4))
    seg_counts = rfm["SegmentLabel"].value_counts()
    seg_counts.plot(kind="bar", ax=ax, color=["#2ecc71", "#3498db", "#e67e22", "#e74c3c"])
    ax.set_title("Customer Segment Distribution")
    ax.set_xlabel("Segment")
    ax.set_ylabel("# Customers")
    ax.bar_label(ax.containers[0], padding=3)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "kmeans_segment_distribution.png"), dpi=150)
    plt.close(fig)

    # 2-D scatter: Recency vs Monetary coloured by segment
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, grp in rfm.groupby("SegmentLabel"):
        ax.scatter(
            grp["Recency"],
            np.log1p(grp["Monetary"]),
            alpha=0.4,
            s=15,
            label=label,
        )
    ax.set_title("Customer Segments: Recency vs log(Monetary)")
    ax.set_xlabel("Recency (days)")
    ax.set_ylabel("log(Monetary)")
    ax.legend(title="Segment")
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "kmeans_segment_scatter.png"), dpi=150)
    plt.close(fig)

    return rfm


# ─────────────────────────────────────────────────────────────────────────────
# 7.  CHURN CLASSIFICATION – RANDOM FOREST
# ─────────────────────────────────────────────────────────────────────────────
def churn_classification(rfm: pd.DataFrame) -> None:
    """
    Binary churn classification:
      Churn = 1  if Recency > 90 days  (customer has not purchased in 3 months)
      Churn = 0  otherwise

    Features: Recency, Frequency, log(Monetary), PrimaryChannel (one-hot)
    Model   : Random Forest Classifier
    """
    print("\n[MODEL] Training churn classification model …")

    # Define churn label
    rfm["Churn"] = (rfm["Recency"] > 90).astype(int)
    print(f"[MODEL] Churn rate: {rfm['Churn'].mean():.1%}")

    # Prepare features
    rfm_feat = rfm.copy()
    rfm_feat["LogMonetary"]  = np.log1p(rfm_feat["Monetary"])
    rfm_feat["LogFrequency"] = np.log1p(rfm_feat["Frequency"])

    # One-hot encode primary channel
    rfm_feat = pd.get_dummies(rfm_feat, columns=["PrimaryChannel"], drop_first=False)

    feature_cols = (
        ["Recency", "LogFrequency", "LogMonetary"]
        + [c for c in rfm_feat.columns if c.startswith("PrimaryChannel_")]
    )

    X = rfm_feat[feature_cols].fillna(0)
    y = rfm_feat["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_SEED, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED, n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("\n[MODEL] Classification Report:\n", classification_report(y_test, y_pred))

    # Confusion matrix
    cm  = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Active", "Churned"])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Churn Classification – Confusion Matrix")
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "churn_confusion_matrix.png"), dpi=150)
    plt.close(fig)

    # Feature importance
    importances = pd.Series(clf.feature_importances_, index=feature_cols).sort_values(
        ascending=False
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    importances.head(10).plot(kind="bar", ax=ax, color="#3498db")
    ax.set_title("Top 10 Feature Importances – Churn Model")
    ax.set_ylabel("Importance")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "churn_feature_importance.png"), dpi=150)
    plt.close(fig)

    print(f"[MODEL] Churn model artefacts saved to ./{OUTPUT_DIR}/")


# ─────────────────────────────────────────────────────────────────────────────
# 8.  EXPORT RESULTS
# ─────────────────────────────────────────────────────────────────────────────
def export_results(rfm: pd.DataFrame) -> None:
    """Save the enriched RFM table to CSV for downstream use (dashboard, reports)."""
    out_path = os.path.join(OUTPUT_DIR, "rfm_segments.csv")
    rfm.to_csv(out_path, index=False)
    print(f"\n[EXPORT] RFM segmented data saved to: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 65)
    print("  OmniCLV – Omnichannel Customer Lifetime Value Optimisation")
    print("  Author : Anushka Yadav")
    print("=" * 65)

    # Step 1 – Load
    df = load_data(DATA_FILE, XLSX_FILE)

    # Step 2 – Clean
    df = clean_data(df)

    # Step 3 – Feature engineering
    df = engineer_features(df)

    # Step 4 – RFM
    rfm = compute_rfm(df)

    # Step 5 – EDA
    run_eda(df, rfm)

    # Step 6 – Segmentation
    rfm = segment_customers_kmeans(rfm)

    # Step 7 – Churn classification
    churn_classification(rfm)

    # Step 8 – Export
    export_results(rfm)

    print("\n[DONE] Pipeline complete. All outputs written to ./" + OUTPUT_DIR + "/")


if __name__ == "__main__":
    main()
