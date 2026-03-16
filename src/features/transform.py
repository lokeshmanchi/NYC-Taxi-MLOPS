"""Feature engineering for NYC Green Taxi fare prediction."""

import os
import json
import warnings
from typing import Tuple

import fsspec
import dask.dataframe as dd

from src.config import config
from src.features.core import BASE_FEATURE_COLS, TARGET_COL
from src.features.summarize import save_summary

# Columns that must be present in the raw parquet before ETL runs.
REQUIRED_RAW_COLS = [
    "lpep_pickup_datetime",
    "fare_amount",
    "trip_distance",
    "passenger_count",
    "PULocationID",
    "DOLocationID",
    "RatecodeID",
]


def validate_raw_data(df: dd.DataFrame) -> None:
    """
    Lightweight pre-flight schema check before ETL runs.

    Checks column presence (metadata-only, free) then samples the first
    partition to catch catastrophic null rates early. Raises ValueError on
    missing columns; emits a warning on >50% nulls in any column.
    """
    missing = [c for c in REQUIRED_RAW_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"ETL aborted — missing required columns: {missing}")

    sample = df[REQUIRED_RAW_COLS].head(1000)  # reads first partition only
    null_rates = sample.isnull().mean()
    critical = null_rates[null_rates > 0.5]
    if not critical.empty:
        warnings.warn(
            f"High null rates detected in sample (>50%): {critical.to_dict()}. "
            "Check source data quality before proceeding.",
            stacklevel=2,
        )


def load_data(data_path: str = "data") -> dd.DataFrame:
    """
    Load data lazily using Dask DataFrame. This allows for out-of-core
    processing on datasets larger than RAM, using a distributed cluster.
    Supports local directories or Cloud Object Storage (s3://, gs://).
    """
    try:
        # dd.read_parquet reads the data lazily, creating a task graph.
        return dd.read_parquet(data_path)
    except Exception as e:
        raise FileNotFoundError(
            f"Could not create Dask DataFrame from {data_path}: {e}"
        )


# `TemporalFeatureEngineer` is defined in `src/features/core.py`.


def filter_outliers(df: dd.DataFrame) -> dd.DataFrame:
    return df[
        (df["fare_amount"] >= 2.5)
        & (df["fare_amount"] <= 250)
        & (df["trip_distance"] > 0.1)
        & (df["trip_distance"] <= 60)
        & (df["passenger_count"] >= 1)
        & (df["passenger_count"] <= 6)
    ]


def engineer_features(df: dd.DataFrame) -> dd.DataFrame:
    # With Dask, these operations build up the task graph.
    # No computation happens yet.
    dt = dd.to_datetime(df["lpep_pickup_datetime"])
    df = df.assign(
        pickup_hour=dt.dt.hour,
        pickup_dayofweek=dt.dt.dayofweek,
        pickup_month=dt.dt.month,
        RatecodeID=df["RatecodeID"].fillna(1).astype(int),
        passenger_count=df["passenger_count"].fillna(1).astype(int),
    )
    return df


def prepare_features(df: dd.DataFrame) -> Tuple[dd.DataFrame, dd.Series]:
    """
    Prepares the raw feature set for training, before pipeline transformations.
    """
    df = filter_outliers(df)
    df = engineer_features(df)
    df = df.dropna(subset=BASE_FEATURE_COLS + [TARGET_COL])
    return df[BASE_FEATURE_COLS], df[TARGET_COL]


def save_processed_data(input_path: str, output_path: str):
    """
    Runs the full feature engineering pipeline and saves the result to Parquet.
    This materializes the lazy Dask graph into files on disk/storage.
    """
    print(f"Processing data from {input_path}...")
    df = load_data(input_path)
    validate_raw_data(df)

    # Apply filters and base feature engineering.
    # Derived features (sin/cos, is_weekend) are NOT saved. They are generated
    # on-the-fly by the training loaders to ensure consistency.
    df = filter_outliers(df)
    df = engineer_features(df)
    df = df.dropna(subset=BASE_FEATURE_COLS + [TARGET_COL])

    final_df = df[BASE_FEATURE_COLS + [TARGET_COL]]

    print(f"Saving processed data to {output_path} with Hive partitioning...")
    # By partitioning on a column like `pickup_month`, we enable downstream
    # consumers to perform "predicate pushdown", reading only the data
    # they need. This is a critical optimization for petabyte-scale lakes.
    final_df.to_parquet(
        output_path,
        write_index=False,
        compression="snappy",
        partition_on=["pickup_month"],
    )

    # Create a manifest file to avoid slow `glob` or `list` operations.
    # This is critical for datasets with millions of partitions.
    fs, fs_path = fsspec.core.url_to_fs(output_path)
    # Use `fs.find` to recursively list all created parquet files.
    all_files = sorted(
        [p for p in fs.find(fs_path, detail=False) if p.endswith(".parquet")]
    )
    manifest_path = os.path.join(output_path, "_manifest.json")
    print(f"Writing manifest for {len(all_files)} files to {manifest_path}...")
    with fs.open(manifest_path, "w") as f:
        json.dump(all_files, f)

    save_summary(output_path)
    print("ETL process complete.")


if __name__ == "__main__":
    # Allow running this script directly to materialize the dataset
    # Usage: python -m src.features.transform
    save_processed_data(config.data_path, config.processed_data_path)
