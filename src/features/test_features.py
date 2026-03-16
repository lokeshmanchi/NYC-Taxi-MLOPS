import pandas as pd
import numpy as np
import dask.dataframe as dd
from src.features.core import TemporalFeatureEngineer
from src.features.transform import filter_outliers


def test_temporal_feature_engineer():
    """Test feature engineering logic (used in both Training and Serving)."""
    # Create sample data (Pandas mode)
    df = pd.DataFrame({
        "pickup_hour": [0, 12, 18],
        "pickup_dayofweek": [0, 5, 6]  # Mon, Sat, Sun
    })
    
    engineer = TemporalFeatureEngineer()
    res = engineer.transform(df)
    
    # Assertions
    assert "hour_sin" in res.columns
    assert "hour_cos" in res.columns
    assert "is_weekend" in res.columns
    assert res.iloc[0]["is_weekend"] == 0  # Monday
    assert res.iloc[1]["is_weekend"] == 1  # Saturday

    # Check cyclic feature correctness: hour=0 → sin=0, cos=1
    assert np.isclose(res.iloc[0]["hour_sin"], 0.0)
    assert np.isclose(res.iloc[0]["hour_cos"], 1.0)


def test_filter_outliers_dask():
    """Test Dask outlier filtering logic (used in ETL)."""
    pdf = pd.DataFrame({
        "fare_amount": [2.5, 100.0, -5.0, 300.0],
        "trip_distance": [1.0, 10.0, 1.0, 1.0],
        "passenger_count": [1, 1, 1, 1]
    })
    ddf = dd.from_pandas(pdf, npartitions=1)
    
    cleaned = filter_outliers(ddf).compute()
    
    # Should drop row 2 (fare -5) and row 3 (fare 300)
    assert len(cleaned) == 2
    assert 2.5 in cleaned["fare_amount"].values