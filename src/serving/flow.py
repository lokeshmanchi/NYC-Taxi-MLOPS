"""
Prefect workflow to orchestrate the ETL and Training pipelines.

This workflow defines the Directed Acyclic Graph (DAG) for a full MLOps run:
1. Run the distributed ETL job to process raw data.
2. Run the distributed training job on the processed data.

To run this flow:
$ prefect server start
$ python -m src.orchestration.flow
"""

import os
import subprocess
from prefect import flow, task
from prefect.tasks import task_input_hash
from datetime import timedelta

from src.features.transform import save_processed_data


@task(
    name="Run Distributed ETL",
    cache_key_fn=task_input_hash,
    cache_expiration=timedelta(days=1),
    retries=2,
    retry_delay_seconds=10,
)
def run_etl(input_path: str, output_path: str):
    """Prefect task to run the Dask-based ETL process."""
    save_processed_data(input_path, output_path)


@task(name="Launch Distributed Training")
def run_training(processed_data_path: str):
    """Prefect task to launch the PyTorch DDP training script."""
    command = "torchrun --standalone --nproc_per_node=2 src/training/train_pytorch_ddp.py"
    print(f"Executing command: {command}")
    # Use subprocess.run for better error handling.
    # If the script fails (non-zero exit code), it will raise an exception.
    subprocess.run(command, shell=True, check=True)


@flow(name="NYC Taxi - ETL & Training Pipeline")
def etl_and_train_flow():
    """Main flow to run ETL and then launch the training job."""
    etl_output_path = os.getenv("PROCESSED_DATA_PATH", "data/processed")
    run_etl(
        input_path=os.getenv("DATA_PATH", "data"), output_path=etl_output_path
    )
    run_training(processed_data_path=etl_output_path)


if __name__ == "__main__":
    etl_and_train_flow()
