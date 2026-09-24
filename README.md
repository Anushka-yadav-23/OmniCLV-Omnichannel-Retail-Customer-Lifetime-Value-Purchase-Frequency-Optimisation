# OmniCLV — Omnichannel Retail Customer Lifetime Value & Purchase Frequency Optimisation

> **Author:** Anushka Yadav

A production-ready Python data science project that analyses transactional retail data to segment customers, predict churn, and surface actionable business insights via an interactive Streamlit dashboard.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset Information](#dataset-information)
3. [Project Structure](#project-structure)
4. [Technologies & Libraries Used](#technologies--libraries-used)
5. [Setup & Installation](#setup--installation)
6. [How to Run](#how-to-run)
7. [Outputs](#outputs)
8. [Dashboard](#dashboard)
9. [Author](#author)

---

## Project Overview

OmniCLV builds an end-to-end analytical pipeline for a UK-based e-commerce retailer:

| Stage | Description |
|-------|-------------|
| **Data Cleaning** | Remove returns, missing Customer IDs, and invalid prices/quantities |
| **Feature Engineering** | Compute `TotalSales = Quantity × UnitPrice`; assign simulated omnichannel labels |
| **RFM Analysis** | Calculate Recency, Frequency, and Monetary metrics per customer |
| **EDA** | Monthly revenue trends, channel breakdowns, geographic distributions |
| **K-Means Clustering** | Segment customers into Champions, Loyal, At-Risk, and Lost/Inactive groups |
| **Churn Classification** | Random Forest model predicting customer churn (Recency > 90 days) |
| **Streamlit Dashboard** | Interactive web UI for business stakeholders |

---

## Dataset Information

| Property | Value |
|----------|-------|
| **File name** | `Online Retail Data Set.csv` (or `.xlsx`) |
| **Source** | [UCI Machine Learning Repository — Online Retail] also available on [Kaggle](https://www.kaggle.com/datasets/siddharththakkar26/online-retail-dataset) |
| **Records** | ~541,909 transactions |
| **Date range** | December 2010 – December 2011 |
| **Columns** | `InvoiceNo`, `StockCode`, `Description`, `Quantity`, `InvoiceDate`, `UnitPrice`, `CustomerID`, `Country` |

> **Setup:** Place `Online Retail Data Set.csv` (preferred) **or** `Online Retail Data Set.xlsx` in the project root before running any script.

---

## Project Structure

```
OmniCLV/
│
├── Online Retail Data Set.csv          # Source dataset (not committed to git)
├── AnushkaYadav_OmniCLV.py             # Main analysis pipeline
├── AnushkaYadav_OmniCLV_Dashboard.py   # Streamlit interactive dashboard
├── requirements.txt                    # Python dependencies
├── README.md                           # This file
│
├── outputs/                            # Auto-created by the pipeline
│   ├── rfm_segments.csv                # Enriched RFM table with segment labels
│   ├── eda_monthly_revenue.png
│   ├── eda_channel_revenue.png
│   ├── eda_rfm_distributions.png
│   ├── eda_top_countries.png
│   ├── kmeans_elbow.png
│   ├── kmeans_segment_distribution.png
│   ├── kmeans_segment_scatter.png
│   ├── churn_confusion_matrix.png
│   └── churn_feature_importance.png
│
└── AnushkaYadav_OmniCLV_ProjectReport.docx   # Full project report
```

---

## Technologies & Libraries Used

| Category | Library / Tool | Version |
|----------|---------------|---------|
| Language | Python | 3.10+ |
| Data Manipulation | Pandas | 2.2.2 |
| Numerical Computing | NumPy | 1.26.4 |
| Machine Learning | Scikit-Learn | 1.4.2 |
| Visualisation | Matplotlib | 3.8.4 |
| Visualisation | Seaborn | 0.13.2 |
| Interactive Charts | Plotly | 5.22.0 |
| Dashboard | Streamlit | 1.35.0 |
| Excel Reading | openpyxl | 3.1.2 |

---

## Setup & Installation

### Prerequisites

- Python 3.10 or later
- pip package manager

### 1 — Clone the Repository

```bash
git clone https://github.com/your-org/omniclv.git
cd omniclv
```

### 2 — Create a Virtual Environment (Recommended)

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

### 4 — Add the Dataset

Copy `Online Retail Data Set.csv` (or `.xlsx`) into the project root directory.

---

## How to Run

### Run the Full Analysis Pipeline

```bash
python AnushkaYadav_OmniCLV.py
```

This executes all pipeline stages in sequence and writes all plots and the RFM CSV to the `outputs/` folder.

Expected console output (abbreviated):

```
=================================================================
  OmniCLV – Omnichannel Customer Lifetime Value Optimisation
  Author : Anushka Yadav
=================================================================
[INFO] Loading data from CSV: Online Retail Data Set.csv
[INFO] Raw shape: (541909, 8)
[CLEAN] Dropped 135080 rows with missing CustomerID
[CLEAN] Dropped 9288 return/cancellation rows
[CLEAN] Final shape after cleaning: (397624, 8)
[RFM] Snapshot date: 2011-12-10
[RFM] Computed RFM for 4,372 unique customers.
[EDA] Plots saved to ./outputs/
[MODEL] Running K-Means clustering (k=4) ...
[MODEL] Training churn classification model ...
[EXPORT] RFM segmented data saved to: outputs/rfm_segments.csv

Pipeline complete. All outputs written to ./outputs/
```

### Launch the Interactive Dashboard

```bash
streamlit run AnushkaYadav_OmniCLV_Dashboard.py
```

Open your browser at `http://localhost:8501` to explore the dashboard.

> **Note:** Run the main pipeline at least once before the dashboard, so that `outputs/rfm_segments.csv` is available.

---

## Outputs

| File | Description |
|------|-------------|
| `outputs/rfm_segments.csv` | Per-customer RFM metrics + K-Means segment labels |
| `outputs/eda_monthly_revenue.png` | Monthly revenue trend line chart |
| `outputs/eda_channel_revenue.png` | Revenue breakdown by sales channel |
| `outputs/eda_rfm_distributions.png` | Histograms of R, F, M distributions |
| `outputs/eda_top_countries.png` | Top 10 countries by revenue |
| `outputs/kmeans_elbow.png` | Elbow curve for k selection |
| `outputs/kmeans_segment_distribution.png` | Customer count per segment |
| `outputs/kmeans_segment_scatter.png` | Recency vs Monetary scatter by segment |
| `outputs/churn_confusion_matrix.png` | RF classifier confusion matrix |
| `outputs/churn_feature_importance.png` | Top 10 feature importances |

---

## Dashboard

The Streamlit dashboard (`AnushkaYadav_OmniCLV_Dashboard.py`) provides:

- **Sidebar Filters** — Sales Channel selector, Date Range picker, Product Category filter
- **KPI Cards** — Total Revenue, Total Transactions, Unique Customers
- **Tab 1 — Revenue Trends** — Monthly revenue line chart + Channel performance bar chart
- **Tab 2 — Customer Segments** — RFM pie chart, Recency vs Monetary scatter, Churn risk histogram

---

## Author

**Anushka Yadav**

Project developed as part of an IBM Data Analytics programme using the UCI Online Retail dataset.

---

*Dataset source: UCI Machine Learning Repository — [Online Retail Data Set](https://archive.ics.uci.edu/ml/datasets/Online+Retail)*
