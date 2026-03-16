"""Feature engineering for NYC Green Taxi fare prediction."""

import os
import json
from typing import Tuple

import fsspec
import numpy as np
import dask.dataframe as dd
from sklearn.base import BaseEstimator, TransformerMixin

# Base features that are read from the source or cleaned.
BASE_FEATURE_COLS = [
    "trip_distance",
    "passenger_count",
    "PULocationID",
    "DOLocationID",
    "pickup_hour",
    "pickup_dayofweek",
    "pickup_month",
    "RatecodeID",
]

# Final feature set after on-the-fly transformation.
FEATURE_COLS = BASE_FEATURE_COLS + ["is_weekend", "hour_sin", "hour_cos"]

TARGET_COL = "fare_amount"


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
        raise FileNotFoundError(f"Could not create Dask DataFrame from {data_path}: {e}")


class TemporalFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer to generate temporal features.
    Ensures identical logic is applied during Training and Serving.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Create a copy to avoid SettingWithCopy warnings on the input df
        X = X.copy()

        # Check for columns before computing to allow robust partial updates
        if "pickup_hour" in X.columns:
            X["hour_sin"] = np.sin(2 * np.pi * X["pickup_hour"] / 24)
            X["hour_cos"] = np.cos(2 * np.pi * X["pickup_hour"] / 24)

        if "pickup_dayofweek" in X.columns:
            X["is_weekend"] = (X["pickup_dayofweek"] >= 5).astype(int)

        return X


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
    df["lpep_pickup_datetime"] = dd.to_datetime(df["lpep_pickup_datetime"])
    df["pickup_hour"] = df["lpep_pickup_datetime"].dt.hour
    df["pickup_dayofweek"] = df["lpep_pickup_datetime"].dt.dayofweek
    df["pickup_month"] = df["lpep_pickup_datetime"].dt.month
    df["RatecodeID"] = df["RatecodeID"].fillna(1).astype(int)
    df["passenger_count"] = df["passenger_count"].fillna(1).astype(int)
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
    all_files = sorted([p for p in fs.find(fs_path, detail=False) if p.endswith(".parquet")])
    manifest_path = os.path.join(output_path, "_manifest.json")
    print(f"Writing manifest for {len(all_files)} files to {manifest_path}...")
    with fs.open(manifest_path, "w") as f:
        json.dump(all_files, f)

    print("ETL process complete.")


if __name__ == "__main__":
    # Allow running this script directly to materialize the dataset
    # Usage: python -m src.features.transform
    INPUT_PATH = os.getenv("DATA_PATH", "data")
    OUTPUT_PATH = os.getenv("PROCESSED_DATA_PATH", "data/processed")
    save_processed_data(INPUT_PATH, OUTPUT_PATH)
