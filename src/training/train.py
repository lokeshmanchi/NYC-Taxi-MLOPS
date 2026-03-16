"""XGBoost training with MLflow experiment tracking."""

import logging
import os
import tempfile

from src.config import config

import mlflow
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Try importing ONNX tools (fault tolerance if packages are missing)
try:
    from skl2onnx import to_onnx
    from onnxruntime.quantization import quantize_dynamic, QuantType
except ImportError:
    to_onnx = None
    quantize_dynamic = None
    print("⚠️ ONNX tools missing. ONNX export will be skipped.")

from src.features.core import (
    BASE_FEATURE_COLS,
    FEATURE_COLS,
    TemporalFeatureEngineer,
)
from src.features.transform import load_data, prepare_features, compute_data_hash, get_data_version

EXPERIMENT_NAME = "nyc-taxi-fare-prediction"

# Setup Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def train(
    data_path: str = "data",
    mlflow_tracking_uri: str = "http://localhost:5000",
) -> str:
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # ── 1. DATA PREPARATION ─────────────────────────────────────────────────
    logger.info(f"Loading data from {data_path}...")
    df = load_data(data_path)
    X, y = prepare_features(df)

    # Materialize Dask to Pandas for local XGBoost training.
    # NOTE: For petabyte scale, use src/training/train_pytorch_ddp.py.
    logger.info("Materializing data into memory (WARNING: High RAM usage)...")
    X, y = X.compute(), y.compute()
    logger.info(f"Dataset: {len(X):,} rows | {len(BASE_FEATURE_COLS)} feats")

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

        # ── 2. MODEL TRAINING ───────────────────────────────────────────────────
        # Wrap the feature engineering and model in a pipeline
        pipeline = Pipeline(
            [
                ("features", TemporalFeatureEngineer()),
                ("model", xgb.XGBRegressor(**params)),
            ]
        )

        pipeline.fit(
            X_train,
            y_train,
            # Pass validation set to the model step specifically
            model__eval_set=[
                (pipeline.named_steps["features"].transform(X_test), y_test)
            ],
            model__verbose=100,
        )

        # ── 3. MODEL EVALUATION ─────────────────────────────────────────────────
        y_pred = pipeline.predict(X_test)
        metrics = {
            "mae": mean_absolute_error(y_test, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "r2": float(r2_score(y_test, y_pred)),
        }
        mlflow.log_metrics(metrics)
        logger.info(
            f"\nMAE: ${metrics['mae']:.2f} RMSE: ${metrics['rmse']:.2f} "
            f"R²: {metrics['r2']:.4f}"
        )

        # ── 4. MODEL LOGGING ────────────────────────────────────────────────────
        # Log the scikit-learn pipeline model to MLflow
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            registered_model_name="nyc-taxi-regressor",  # Optional
        )

        # ── 5. ONNX EXPORT (Edge/Cloud Optimization) ────────────────────────────
        if to_onnx:
            try:
                logger.info("Converting XGBoost model to ONNX for edge...")
                # We convert ONLY the XGBoost estimator. Feature engineering remains
                # in Python to avoid complex ONNX custom ops, creating a "Hybrid"
                # inference pattern suitable for robotics.
                xgb_s = pipeline.named_steps["model"]
                feat_s = pipeline.named_steps["features"]

                # Transform a small sample to get the exact input shape/types
                X_sample = X_test.iloc[:5]
                X_trans = feat_s.transform(X_sample)
                # Ensure float32 inputs for broad hardware compatibility
                X_tensor = X_trans[FEATURE_COLS].astype(np.float32)

                # Convert to ONNX
                onx = to_onnx(xgb_s, X_tensor)

                # Save and log — use a temp dir so concurrent runs don't collide
                with tempfile.TemporaryDirectory() as tmp:
                    onnx_path = os.path.join(tmp, "model.onnx")
                    with open(onnx_path, "wb") as f:
                        f.write(onx.SerializeToString())

                    mlflow.log_artifact(onnx_path, artifact_path="onnx")
                    logger.info("✅ ONNX saved & logged")

                    # INT8 quantization — ~4x smaller, 2-3x faster on edge hardware.
                    # Uses dynamic quantization (no calibration dataset required).
                    if quantize_dynamic is not None:
                        int8_path = os.path.join(tmp, "model_int8.onnx")
                        quantize_dynamic(onnx_path, int8_path, weight_type=QuantType.QUInt8)
                        mlflow.log_artifact(int8_path, artifact_path="onnx")
                        logger.info("✅ INT8 ONNX saved & logged")
                    # TemporaryDirectory context manager cleans up both files automatically
            except Exception as e:
                logger.error(f"❌ Failed to export ONNX model: {e}")

        # Log data provenance — links this run to its exact input data snapshot
        raw_data_hash = compute_data_hash(data_path)
        mlflow.set_tag("data_raw_hash", raw_data_hash)

        processed_version = get_data_version()
        if processed_version.get("snapshot_hash"):
            mlflow.set_tag("data_processed_hash", processed_version["snapshot_hash"])
            mlflow.set_tag("data_created_at", processed_version.get("created_at", "unknown"))
            mlflow.set_tag("data_file_count", str(processed_version.get("file_count", "")))

        run_id = run.info.run_id
        logger.info(f"MLflow run_id: {run_id}")

    return run_id


if __name__ == "__main__":
    train(
        data_path=config.data_path,
        mlflow_tracking_uri=config.mlflow_tracking_uri,
    )
