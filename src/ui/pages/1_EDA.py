"""EDA page — exploratory data analysis of NYC Green Taxi dataset."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# At petabyte scale, the EDA dashboard cannot load raw data. Instead, it loads
# small, pre-aggregated summary files created by a distributed processing job
# (e.g., using Spark or Dask).
HOURLY_SUMMARY_PATH = "data/summary/hourly_summary.parquet"

st.set_page_config(page_title="EDA — NYC Taxi", page_icon="📊", layout="wide")
st.title("📊 Exploratory Data Analysis")


@st.cache_data
def load_summary_data(data_path: str) -> pd.DataFrame:
    """Loads a pre-aggregated summary DataFrame."""
    try:
        # In a real scenario, this file would be the output of a daily/hourly
        # Spark/Dask job that aggregates petabytes of raw data.
        return pd.read_parquet(data_path)
    except FileNotFoundError:
        st.error(f"Summary file not found at `{data_path}`. Run the aggregation pipeline first.")
        return None


hourly_summary = load_summary_data(HOURLY_SUMMARY_PATH)

if hourly_summary is None:
    st.stop()

st.info("Showing pre-aggregated summaries. At petabyte scale, live EDA on raw data is not feasible.")

st.subheader("Average Fare and Trip Count by Hour")
st.caption("This data is loaded from a pre-aggregated summary file.")
fig = go.Figure()
fig.add_trace(go.Bar(x=hourly_summary["pickup_hour"], y=hourly_summary["trip_count"], name="Trip Count", yaxis="y"))
fig.add_trace(go.Scatter(x=hourly_summary["pickup_hour"], y=hourly_summary["avg_fare"], name="Avg Fare ($)", yaxis="y2", mode="lines+markers"))
fig.update_layout(
    xaxis_title="Hour of Day",
    yaxis=dict(title="Total Trips"),
    yaxis2=dict(title="Avg Fare ($)", overlaying="y", side="right"),
    height=350,
    legend=dict(x=0, y=1),
)
st.plotly_chart(fig, use_container_width=True)
