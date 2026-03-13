"""Home page — NYC Taxi MLOps Dashboard."""

import os

import pandas as pd
import streamlit as st

DATA_PATH = os.getenv("DATA_PATH", "data")

st.set_page_config(
    page_title="NYC Taxi MLOps",
    page_icon="🚕",
    layout="wide",
)

st.title("🚕 NYC Green Taxi — MLOps Dashboard")
st.markdown("**Dataset:** Jan–May 2025 · **Model:** XGBoost fare prediction")

st.markdown("---")

# Load summary stats
@st.cache_data
def load_summary(data_path: str):
    import glob
    files = sorted(glob.glob(f"{data_path}/green_tripdata_*.parquet"))
    dfs = [pd.read_parquet(f, columns=["fare_amount", "trip_distance", "passenger_count", "lpep_pickup_datetime"]) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df = df[(df["fare_amount"] >= 2.5) & (df["trip_distance"] > 0)]
    return df, len(files)

try:
    df, n_files = load_summary(DATA_PATH)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Trips", f"{len(df):,}")
    col2.metric("Avg Fare", f"${df['fare_amount'].mean():.2f}")
    col3.metric("Avg Distance", f"{df['trip_distance'].mean():.2f} mi")
    col4.metric("Months Loaded", str(n_files))

except Exception as e:
    st.error(f"Could not load data from `{DATA_PATH}`: {e}")

st.markdown("---")
st.markdown("""
### Navigation
| Page | Description |
|------|-------------|
| **EDA** | Explore fare distributions, trip patterns by hour/day, zone analysis |
| **Model Performance** | MLflow metrics, feature importance, actual vs predicted |
| **Live Prediction** | Enter trip details and get a real-time fare estimate |
""")

st.markdown("---")
st.markdown("""
### Services
- **MLflow UI**: [http://localhost:5000](http://localhost:5000)
- **Prefect UI**: [http://localhost:4200](http://localhost:4200)
- **API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
""")
