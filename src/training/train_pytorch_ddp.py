"""
Hypothetical training script using PyTorch DistributedDataParallel (DDP).

This script demonstrates how to train a PyTorch model on the NYC Taxi dataset
in a distributed fashion, ready for multi-GPU or multi-node scaling.

To run on a machine with 4 GPUs:
torchrun --standalone --nproc_per_node=4 src/training/train_pytorch_ddp.py
"""

import os

import fsspec
import json
import mlflow
import torch
import torch.distributed as dist
import torch.nn as nn
import pandas as pd
import pyarrow.parquet as pq
import pyarrow.dataset as ds
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from src.features.transform import TemporalFeatureEngineer, FEATURE_COLS, BASE_FEATURE_COLS



def setup_ddp():
    """Initializes the distributed environment from environment variables."""
    dist.init_process_group(backend="nccl")  # "nccl" for GPU, "gloo" for CPU
    # `torchrun` sets RANK, LOCAL_RANK, and WORLD_SIZE env vars.
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    print(f"Starting DDP on rank {rank} (local rank {local_rank}) of {world_size} processes.")
    return rank, local_rank, world_size


def cleanup_ddp():
    """Cleans up the distributed environment."""
    dist.destroy_process_group()


class TabularNet(nn.Module):
    """A simple MLP for tabular regression."""

    def __init__(self, num_features: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.model(x)


class ParquetStreamingDataset(IterableDataset):
    """
    Streams data from Parquet files directly, without loading the full dataset into RAM.
    Handles sharding for Distributed Data Parallel (DDP).
    """
    def __init__(self, data_path: str, rank: int, world_size: int, batch_size: int = 4096):
        self.data_path = data_path
        self.batch_size = batch_size
        self.feature_engineer = TemporalFeatureEngineer()
        self.columns = BASE_FEATURE_COLS + [TARGET_COL]

        # Read the manifest file to get a list of all data partitions.
        # This avoids a very slow and expensive listing operation on object storage.
        manifest_path = os.path.join(self.data_path, "_manifest.json")
        fs, _ = fsspec.core.url_to_fs(manifest_path)
        with fs.open(manifest_path, "r") as f:
            all_files = json.load(f)

        # DDP Sharding: Assign a unique subset of files to this specific GPU (rank).
        self.my_files = all_files[rank::world_size]

    def __iter__(self):
        worker_info = get_worker_info()
        files_to_process = self.my_files
        
        # If using multiple workers per GPU, split the files again among workers
        if worker_info is not None:
            files_to_process = self.my_files[worker_info.id::worker_info.num_workers]

        for file_path in files_to_process:
            # Use pyarrow to stream batches with column projection.
            parquet_file = pq.ParquetFile(file_path, filesystem=fsspec.core.url_to_fs(file_path)[0])
            for batch in parquet_file.iter_batches(batch_size=self.batch_size, columns=self.columns):
                df_raw = batch.to_pandas()
                
                # Apply on-the-fly feature engineering to ensure consistency
                df_transformed = self.feature_engineer.transform(df_raw)
                
                # The ETL job saved BASE_FEATURE_COLS + TARGET_COL.
                X = df_transformed[FEATURE_COLS].values
                y = df_raw["fare_amount"].values

                # Yield a full batch at once. 
                # We set batch_size=None in DataLoader to handle this correctly.
                yield torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32).view(-1, 1)


def run_training(data_path: str, model_output_path: str):
    """Main DDP training function."""
    # Configuration from env vars for tuning at scale
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "4096"))
    NUM_WORKERS = int(os.getenv("NUM_WORKERS", "4"))
    GRAD_ACCUMULATION_STEPS = int(os.getenv("GRAD_ACCUMULATION_STEPS", "4"))
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    EXPERIMENT_NAME = "nyc-taxi-fare-prediction-ddp"

    rank, local_rank, world_size = setup_ddp()

    try:
        # 1. DATA PREPARATION
        # Initialize the streaming dataset. No data is loaded into RAM yet.
        dataset = ParquetStreamingDataset(data_path, rank=rank, world_size=world_size, batch_size=BATCH_SIZE)

        # Set up MLflow tracking only on the master process
        if rank == 0:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            mlflow.set_experiment(EXPERIMENT_NAME)
            mlflow.start_run()
            params = {
                "batch_size": BATCH_SIZE,
                "num_workers": NUM_WORKERS,
                "grad_accumulation_steps": GRAD_ACCUMULATION_STEPS,
                "world_size": world_size,
                "base_features": BASE_FEATURE_COLS,
            }
            mlflow.log_params(params)
        
        # batch_size=None because the Dataset yields pre-batched tensors.
        dataloader = DataLoader(dataset, batch_size=None, num_workers=NUM_WORKERS, pin_memory=True)

        # 2. MODEL INITIALIZATION
        # The model is created on the CPU, then moved to its assigned GPU.
        model = TabularNet(num_features=len(FEATURE_COLS)).to(local_rank)

        # DDP wraps the model and handles gradient synchronization across all GPUs.
        # From now on, you only use `ddp_model`.
        ddp_model = DDP(model, device_ids=[local_rank])

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(ddp_model.parameters(), lr=0.001)

        # 2.1 ELASTIC CHECKPOINTING
        # Load from S3/local checkpoint if it exists to recover from failures
        checkpoint_dir = os.path.dirname(model_output_path)
        checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pt")
        fs, cp_fs_path = fsspec.core.url_to_fs(checkpoint_path)
        start_epoch = 0

        if fs.exists(cp_fs_path):
            if rank == 0:
                print(f"Resuming from checkpoint: {checkpoint_path}")
            # Open stream with fsspec to support S3
            with fs.open(cp_fs_path, "rb") as f:
                # map_location is crucial for DDP to load onto correct device
                checkpoint = torch.load(f, map_location=f"cuda:{local_rank}")
            ddp_model.module.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"] + 1

        # 3. TRAINING LOOP
        for epoch in range(start_epoch, 5):
            # Set model to training mode
            ddp_model.train()
            for i, (inputs, targets) in enumerate(dataloader):
                inputs = inputs.to(local_rank)
                targets = targets.to(local_rank)

                outputs = ddp_model(inputs)
                loss = criterion(outputs, targets)
                
                # Scale the loss for gradient accumulation
                loss = loss / GRAD_ACCUMULATION_STEPS
                loss.backward()  # Gradients are computed on each GPU.

                if (i + 1) % GRAD_ACCUMULATION_STEPS == 0:
                    optimizer.step() # Update weights
                    optimizer.zero_grad() # Reset gradients
            # Log loss from the master process only to avoid spamming logs.
            epoch_loss = loss.item() * GRAD_ACCUMULATION_STEPS
            if rank == 0:
                print(f"Epoch {epoch+1}/{5}, Loss: {epoch_loss:.4f}")
                # Log metrics to MLflow
                mlflow.log_metric("loss", epoch_loss, step=epoch)
                
                # Save checkpoint at the end of every epoch
                print(f"Saving checkpoint to {checkpoint_path}")
                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": ddp_model.module.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                }
                with fs.open(cp_fs_path, "wb") as f:
                    torch.save(checkpoint, f)

        # 4. SAVE MODEL
        # Only the master process (rank 0) should save the model to prevent race conditions.
        if rank == 0:
            # When saving a DDP model, you save the underlying `module`'s state_dict.
            # Log the final model to MLflow
            mlflow.pytorch.log_model(
                pytorch_model=ddp_model.module,
                artifact_path="model",
                registered_model_name="nyc-taxi-regressor-ddp"
            )
            print(f"Model saved to {model_output_path}")
            mlflow.end_run()

    finally:
        # Ensure cleanup happens even if training crashes
        cleanup_ddp()


if __name__ == "__main__":
    run_training(
        # Point to the PROCESSED data (the output of the ETL job)
        data_path=os.getenv("PROCESSED_DATA_PATH", "data/processed"),
        model_output_path=os.getenv("MODEL_OUTPUT_PATH", "models/pytorch_model.pt"),
    )