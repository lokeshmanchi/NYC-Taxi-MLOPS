"""XGBoost training with MLflow experiment tracking."""

import os

import mlflow
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.features.transform import BASE_FEATURE_COLS, FEATURE_COLS, load_data, prepare_features, TemporalFeatureEngineer

EXPERIMENT_NAME = "nyc-taxi-fare-prediction"


def train(
    data_path: str = "data",
    mlflow_tracking_uri: str = "http://localhost:5000",
) -> str:
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    print(f"Loading data from {data_path}...")
    df = load_data(data_path)
    X, y = prepare_features(df)
    
    # Materialize Dask to Pandas for local XGBoost training.
    # NOTE: For petabyte scale, use src/training/train_pytorch_ddp.py instead.
    print("Materializing data into memory...")
    X, y = X.compute(), y.compute()
    print(f"Dataset: {len(X):,} rows after filtering, {len(BASE_FEATURE_COLS)} base features")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    params = {
        "n_estimators": 400,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "random_state": 42,
        "n_jobs": -1,
        "objective": "reg:squarederror",
    }

    with mlflow.start_run() as run:
        mlflow.log_params(params)
        mlflow.log_param("train_rows", len(X_train))
        mlflow.log_param("test_rows", len(X_test))
        mlflow.log_param("features", FEATURE_COLS)

        # Wrap the feature engineering and model in a pipeline
        pipeline = Pipeline([
            ("features", TemporalFeatureEngineer()),
            ("model", xgb.XGBRegressor(**params))
        ])
        
        pipeline.fit(
            X_train,
            y_train,
            # Pass validation set to the model step specifically
            model__eval_set=[(pipeline.named_steps["features"].transform(X_test), y_test)],
            model__verbose=100,
        )

        y_pred = pipeline.predict(X_test)
        metrics = {
            "mae": float(mean_absolute_error(y_test, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "r2": float(r2_score(y_test, y_pred)),
        }
        mlflow.log_metrics(metrics)
        print(f"\nMetrics — MAE: ${metrics['mae']:.2f}  RMSE: ${metrics['rmse']:.2f}  R²: {metrics['r2']:.4f}")

        # Log the scikit-learn pipeline model to MLflow
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            registered_model_name="nyc-taxi-regressor" # Optional: register the model
        )

        run_id = run.info.run_id
        print(f"MLflow run_id: {run_id}")

    return run_id


if __name__ == "__main__":
    train(
        data_path=os.getenv("DATA_PATH", "data"),
        mlflow_tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"),
    )
