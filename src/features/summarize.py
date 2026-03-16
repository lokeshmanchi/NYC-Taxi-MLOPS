"""Generate pre-aggregated summary files for the EDA dashboard."""

import os

import dask.dataframe as dd
import pandas as pd

from src.config import config
from src.features.core import TARGET_COL

SUMMARY_PATH = "data/summary"


def generate_hourly_summary(df: dd.DataFrame) -> pd.DataFrame:
    """Group by pickup_hour, return trip_count and avg_fare as a Pandas DataFrame."""
    return (
        df.groupby("pickup_hour")[TARGET_COL]
        .agg(["count", "mean"])
        .rename(columns={"count": "trip_count", "mean": "avg_fare"})
        .compute()
        .reset_index()
        .sort_values("pickup_hour")
    )


def save_summary(processed_data_path: str, summary_path: str = SUMMARY_PATH):
    """Load processed Parquet, compute hourly summary, write to data/summary/."""
    print(f"Generating summary from {processed_data_path}...")
    df = dd.read_parquet(processed_data_path)
    summary = generate_hourly_summary(df)

    os.makedirs(summary_path, exist_ok=True)
    out = os.path.join(summary_path, "hourly_summary.parquet")
    summary.to_parquet(out, index=False)
    print(f"Summary saved to {out} ({len(summary)} rows).")


if __name__ == "__main__":
    save_summary(config.processed_data_path)
