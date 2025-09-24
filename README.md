# Auto-Adaptive Dashboard (Streamlit)

A turnkey Streamlit web app that lets you upload **CSV/Excel**, infers the dataset, and renders a **custom dashboard** with KPIs, charts, and filters.

## ▶️ Quickstart

```bash
# 1) Create & activate a virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Run the app
streamlit run app.py
```

Then open the local URL shown in the terminal.

## Features
- Drag-and-drop upload for .csv/.xlsx
- Schema inference (numeric, categorical, dates)
- KPI cards: Total Revenue, Average ARPU, Churn Rate, Customer Count (auto-detected)
- Suggested charts (adaptive to your columns)
- Sidebar filters (categorical multiselects, date range)
- Data profile (describe(), categorical cardinality, missing values)
- Download filtered data

## Demo dataset
A demo file `telecom_dashboard_dataset.xlsx` is bundled. Click **"Use demo telecom dataset"** in the sidebar to load it instantly.

## Notes
- The app adapts to **any** dataset. It guesses common KPI names but still works even if those KPIs are absent.
- Charts use Plotly for interactivity.
