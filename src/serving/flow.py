"""
Prefect workflow to orchestrate the ETL and Training pipelines.

This workflow defines the Directed Acyclic Graph (DAG) for a full MLOps run:
1. Run the distributed ETL job to process raw data.
2. Run the distributed training job on the processed data.

To run this flow:
$ prefect server start
$ python -m src.orchestration.flow
"""

import subprocess
from datetime import timedelta

from prefect import flow, task
from prefect.tasks import task_input_hash

from src.config import config
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
def run_training():
    """Prefect task to launch the PyTorch DDP training script."""
    nproc = config.nproc_per_node
    rdzv_endpoint = config.rdzv_endpoint
    rdzv_id = config.rdzv_id

    if rdzv_endpoint:
        # Multi-node: explicit rendezvous — do NOT use --standalone (mutually exclusive)
        command = (
            f"torchrun --nproc_per_node={nproc}"
            f" --rdzv_id {rdzv_id} --rdzv_backend c10d"
            f" --rdzv_endpoint {rdzv_endpoint}"
            f" src/training/train_pytorch_ddp.py"
        )
    else:
        # Single-node: --standalone handles rendezvous automatically
        command = f"torchrun --standalone --nproc_per_node={nproc} src/training/train_pytorch_ddp.py"

    print(f"Executing command: {command}")
    subprocess.run(command, shell=True, check=True)


@flow(name="NYC Taxi - ETL & Training Pipeline")
def etl_and_train_flow():
    """Main flow to run ETL and then launch the training job."""
    etl_output_path = config.processed_data_path
    run_etl(input_path=config.data_path, output_path=etl_output_path)
    run_training()


if __name__ == "__main__":
    etl_and_train_flow()
