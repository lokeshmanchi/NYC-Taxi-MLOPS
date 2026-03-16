"""Core feature definitions and light-weight engineering for model parity."""

import numpy as np
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


class TemporalFeatureEngineer(BaseEstimator, TransformerMixin):
    """Scikit-learn-compatible transformer to generate temporal features."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        if "pickup_hour" in X.columns:
            X["hour_sin"] = np.sin(2 * np.pi * X["pickup_hour"] / 24)
            X["hour_cos"] = np.cos(2 * np.pi * X["pickup_hour"] / 24)

        if "pickup_dayofweek" in X.columns:
            X["is_weekend"] = (X["pickup_dayofweek"] >= 5).astype(int)

        return X
