"""Prefect training pipeline for NYC Taxi fare prediction."""

from typing import Optional

from prefect import flow, task
from prefect.logging import get_run_logger
from src.config import config


@task(name="train-xgboost", retries=1, retry_delay_seconds=30)
def train_task(data_path: str, mlflow_tracking_uri: str) -> str:
    from src.training.train import train

    logger = get_run_logger()
    logger.info(f"Starting training — data: {data_path}")
    run_id = train(
        data_path=data_path,
        mlflow_tracking_uri=mlflow_tracking_uri,
    )
    logger.info(f"Training complete. MLflow run_id={run_id}")
    return run_id


@flow(name="nyc-taxi-training-pipeline", log_prints=True)
def training_pipeline(
    data_path: Optional[str] = None,
    mlflow_tracking_uri: Optional[str] = None,
) -> str:
    data_path = data_path or config.data_path
    mlflow_tracking_uri = mlflow_tracking_uri or config.mlflow_tracking_uri

    run_id = train_task(data_path, mlflow_tracking_uri)
    return run_id


if __name__ == "__main__":
    training_pipeline()
