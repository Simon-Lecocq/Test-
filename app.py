import io
import json
from typing import Dict, List, Optional

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Auto Dashboard", layout="wide")

st.title("📊 Auto-Adaptive Dashboard")
st.caption("Upload a CSV or Excel file. The app infers schema, builds KPIs and charts automatically.")

# ---------- Utils ----------
def load_file(uploaded_file) -> pd.DataFrame:
    if uploaded_file is None:
        return pd.DataFrame()
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    elif name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded_file)
    else:
        st.error("Unsupported file type. Please upload .csv or .xlsx")
        return pd.DataFrame()

def sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {c: c.strip().replace(" ", "_").replace("(", "").replace(")", "").replace("€","EUR").replace("/", "_").replace("-", "_") for c in df.columns}
    return df.rename(columns=mapping)

def infer_types(df: pd.DataFrame):
    date_cols, numeric_cols = [], []
    for c in df.columns:
        if any(tok in c.lower() for tok in ["date", "start", "end"]):
            try:
                df[c] = pd.to_datetime(df[c], errors="coerce")
                date_cols.append(c)
            except:
                pass
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in df.columns if c not in numeric_cols + date_cols]
    return date_cols, numeric_cols, categorical_cols

def kpis(df: pd.DataFrame) -> Dict[str, float]:
    # Try common KPI names with fallbacks
    revenue_col = next((c for c in df.columns if c.lower().startswith("revenue_ytd")), None)
    arpu_col = next((c for c in df.columns if c.lower().startswith("arpu")), None)
    churn_col = next((c for c in df.columns if c.lower() == "churn_flag"), None)

    total_revenue = float(df[revenue_col].sum()) if revenue_col else float("nan")
    average_arpu = float(df[arpu_col].mean()) if arpu_col else float("nan")
    churn_rate = float(df[churn_col].mean()) if churn_col else float("nan")
    customer_count = int(df.shape[0])

    return {
        "Total Revenue (YTD)": total_revenue,
        "Average ARPU": average_arpu,
        "Churn Rate": churn_rate,
        "Customer Count": customer_count
    }

def suggest_charts(df: pd.DataFrame, date_cols, numeric_cols, categorical_cols):
    charts = []
    # Revenue by Region
    revenue_col = next((c for c in df.columns if c.lower().startswith("revenue_ytd")), None)
    if revenue_col and "Region" in df.columns:
        charts.append({"title": "Revenue by Region", "type": "bar", "x": "Region", "y": revenue_col, "agg": "sum"})
    # Churn by Segment
    if "Churn_Flag" in df.columns and "Segment" in df.columns:
        charts.append({"title": "Churn Rate by Segment", "type": "bar", "x": "Segment", "y": "Churn_Flag", "agg": "mean"})
    # ARPU by Product
    arpu_col = next((c for c in df.columns if c.lower().startswith("arpu")), None)
    if arpu_col and "Product_Type" in df.columns:
        charts.append({"title": "ARPU by Product Type", "type": "bar", "x": "Product_Type", "y": arpu_col, "agg": "mean"})
    # Satisfaction vs Tickets
    if "Satisfaction_Score" in df.columns and "Support_Tickets" in df.columns:
        charts.append({"title": "Satisfaction vs Support Tickets", "type": "scatter", "x": "Support_Tickets", "y": "Satisfaction_Score", "agg": None})
    # Generic: Top category counts for first categorical column
    if categorical_cols:
        c = categorical_cols[0]
        charts.append({"title": f"Top values of {c}", "type": "bar", "x": c, "y": None, "agg": "count"})
    return charts

def apply_filters(df, categorical_cols, date_cols):
    with st.sidebar:
        st.header("🔎 Filters")
        filters = {}
        # Categorical multiselects
        for c in categorical_cols:
            nunique = df[c].nunique(dropna=True)
            if 1 < nunique <= 30:
                vals = sorted([v for v in df[c].dropna().unique().tolist()])
                sel = st.multiselect(f"{c}", vals, default=vals)
                if sel:
                    filters[c] = sel
        # Date range (first date col only)
        if date_cols:
            c = date_cols[0]
            min_d = pd.to_datetime(df[c]).min()
            max_d = pd.to_datetime(df[c]).max()
            if pd.notna(min_d) and pd.notna(max_d):
                start, end = st.date_input("Date range", value=(min_d.date(), max_d.date()))
                if start and end:
                    filters[c] = (pd.to_datetime(start), pd.to_datetime(end))

    # Apply filters
    out = df.copy()
    for k, v in filters.items():
        if k in date_cols:
            start, end = v
            out = out[(out[k] >= start) & (out[k] <= end)]
        else:
            out = out[out[k].isin(v)]
    return out

def render_chart(df, spec):
    t = spec["type"]
    if t == "bar":
        x = spec["x"]
        y = spec["y"]
        if spec["agg"] == "count" or y is None:
            plot_df = df.groupby(x).size().reset_index(name="count")
            fig = px.bar(plot_df, x=x, y="count")
        elif spec["agg"] == "sum":
            plot_df = df.groupby(x)[y].sum().reset_index()
            fig = px.bar(plot_df, x=x, y=y)
        elif spec["agg"] == "mean":
            plot_df = df.groupby(x)[y].mean().reset_index()
            fig = px.bar(plot_df, x=x, y=y)
        else:
            st.warning(f"Unsupported agg for bar: {spec['agg']}")
            return
        st.plotly_chart(fig, use_container_width=True)

    elif t == "scatter":
        x, y = spec["x"], spec["y"]
        fig = px.scatter(df, x=x, y=y, trendline="ols")
        st.plotly_chart(fig, use_container_width=True)

    elif t == "line":
        x, y = spec["x"], spec["y"]
        fig = px.line(df, x=x, y=y)
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info(f"Unknown chart type: {t}")

# ---------- App flow ----------

st.sidebar.write("💾 Upload your dataset")
uploaded = st.sidebar.file_uploader("CSV or Excel", type=["csv", "xlsx", "xls"])

# Optional demo dataset button
use_demo = st.sidebar.button("Use demo telecom dataset")
if use_demo:
    demo_path = "telecom_dashboard_dataset.xlsx"
    # Load bundled demo
    try:
        df_demo = pd.read_excel(demo_path)
        uploaded = io.BytesIO()
        df_demo.to_excel(uploaded, index=False)
        uploaded.seek(0)
        uploaded.name = "telecom_dashboard_dataset.xlsx"
    except Exception as e:
        st.sidebar.error(f"Demo failed: {e}")

df = load_file(uploaded)
if df.empty:
    st.info("Upload a file to begin. Supported: .csv, .xlsx")
    st.stop()

df = sanitize_columns(df)
date_cols, numeric_cols, categorical_cols = infer_types(df)

# Filters
filtered = apply_filters(df, categorical_cols, date_cols)

# KPIs
kpi_vals = kpis(filtered)
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Revenue (YTD)", f"{kpi_vals['Total Revenue (YTD)']:.2f}" if pd.notna(kpi_vals['Total Revenue (YTD)']) else "—")
k2.metric("Average ARPU", f"{kpi_vals['Average ARPU']:.2f}" if pd.notna(kpi_vals['Average ARPU']) else "—")
k3.metric("Churn Rate", f"{kpi_vals['Churn Rate']:.2%}" if pd.notna(kpi_vals['Churn Rate']) else "—")
k4.metric("Customer Count", f"{kpi_vals['Customer Count']}")

# Suggested charts
charts = suggest_charts(filtered, date_cols, numeric_cols, categorical_cols)
st.subheader("📈 Suggested Charts")
for spec in charts:
    with st.expander(spec["title"], expanded=True):
        render_chart(filtered, spec)

# Data profile
st.subheader("🔬 Data Profile")
c1, c2 = st.columns(2)
with c1:
    st.markdown("**Numeric Summary**")
    if numeric_cols:
        st.dataframe(filtered[numeric_cols].describe().T)
    else:
        st.write("No numeric columns detected.")
with c2:
    st.markdown("**Categorical Cardinalities**")
    card = pd.DataFrame({"column": categorical_cols, "unique_values": [filtered[c].nunique(dropna=True) for c in categorical_cols]})
    st.dataframe(card)

st.subheader("🧹 Missing Values")
na = filtered.isna().mean().sort_values(ascending=False)
st.bar_chart(na)

st.subheader("🔎 Data Preview")
st.dataframe(filtered, use_container_width=True)

# Download filtered data
out_csv = filtered.to_csv(index=False).encode("utf-8")
st.download_button("Download filtered CSV", out_csv, file_name="filtered_data.csv", mime="text/csv")
