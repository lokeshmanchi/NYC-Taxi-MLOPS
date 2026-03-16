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
import pyarrow.parquet as pq
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from torch.cuda.amp import GradScaler

from src.features.core import (
    TemporalFeatureEngineer,
    FEATURE_COLS,
    BASE_FEATURE_COLS,
    TARGET_COL,
)
from src.config import config


def setup_ddp():
    """Initializes the distributed environment from environment variables."""
    backend = config.ddp_backend
    try:
        dist.init_process_group(backend=backend)
    except Exception as e:
        if backend == "nccl":
            print(
                "Warning: NCCL init failed, falling back to gloo (CPU or mixed).",
                e
            )
            dist.init_process_group(backend="gloo")
            backend = "gloo"
        else:
            raise

    # `torchrun` sets RANK, LOCAL_RANK, and WORLD_SIZE env vars.
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))

    # Set GPU device only when using GPU backend.
    if backend == "nccl":
        torch.cuda.set_device(local_rank)
        torch.backends.cudnn.benchmark = True

    print(f"Start DDP rank {rank} (local {local_rank}) of {world_size}, backend={backend}.")
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

    def __init__(
        self,
        data_path: str,
        rank: int,
        world_size: int,
        batch_size: int = 4096,
    ):
        self.data_path = data_path
        self.batch_size = batch_size
        self.feature_engineer = TemporalFeatureEngineer()
        self.columns = BASE_FEATURE_COLS + [TARGET_COL]

        # Read the manifest to get a list of all data partitions.
        # Avoids expensive listing on object storage.
        manifest_path = os.path.join(self.data_path, "_manifest.json")
        fs, fs_path = fsspec.core.url_to_fs(manifest_path)

        if fs.exists(fs_path):
            with fs.open(manifest_path, "r") as f:
                all_files = json.load(f)
        else:
            print(f"Manifest file not found at {manifest_path}, using fs.find() fallback.")
            all_files = sorted(fs.find(self.data_path, detail=False))

        # DDP Sharding: Assign subset of files to this GPU (rank).
        self.my_files = all_files[rank::world_size]

    def __iter__(self):
        worker_info = get_worker_info()
        files_to_process = self.my_files

        # If using multiple workers per GPU, split the files again among workers
        if worker_info:
            wid = worker_info.id
            wnum = worker_info.num_workers
            files_to_process = self.my_files[wid::wnum]

        # Initialize filesystem INSIDE the worker process.
        # S3/GCS connections are often not fork-safe and shouldn't be shared across workers.
        # 'use_listings_cache=False' prevents stale metadata issues in long-running jobs.
        fs, _ = fsspec.core.url_to_fs(self.data_path, use_listings_cache=False)

        for file_path in files_to_process:
            # Use pyarrow to stream batches with column projection.
            parquet_file = pq.ParquetFile(file_path, filesystem=fs)
            iter_b = parquet_file.iter_batches(batch_size=self.batch_size, columns=self.columns)
            for batch in iter_b:
                df_raw = batch.to_pandas()

                # Apply on-the-fly feature engineering to ensure consistency
                df_transformed = self.feature_engineer.transform(df_raw)

                # The ETL job saved BASE_FEATURE_COLS + TARGET_COL.
                X = df_transformed[FEATURE_COLS].values
                y = df_raw["fare_amount"].values

                # Yield a full batch at once.
                # We set batch_size=None in DataLoader to handle this correctly.
                yield (
                    torch.tensor(X, dtype=torch.float32),
                    torch.tensor(y, dtype=torch.float32).view(-1, 1),
                )


def run_training(data_path: str, model_output_path: str):
    """Main DDP training function."""
    # Configuration from centralized config
    BATCH_SIZE = config.batch_size
    NUM_WORKERS = config.num_workers
    GRAD_ACCUMULATION_STEPS = config.grad_accumulation_steps
    EPOCHS = config.epochs
    LOG_STEP_INTERVAL = config.log_step_interval
    MLFLOW_TRACKING_URI = config.mlflow_tracking_uri
    EXPERIMENT_NAME = config.experiment_name + "-ddp"

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

        # "Identify and remediate bottlenecks... optimize throughput"
        # Using torch.compile (PyTorch 2.0+) fuses layers and optimizes CUDA
        # kernels for the GPU architecture (e.g., A100/H100).
        # We compile BEFORE wrapping in DDP.
        print("Compiling model with torch.compile for higher throughput...")
        model = torch.compile(model)

        # DDP wraps the model and handles gradient synchronization across all GPUs.
        # From now on, you only use `ddp_model`.
        ddp_model = DDP(model, device_ids=[local_rank])

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(ddp_model.parameters(), lr=0.001)
        # Initialize GradScaler for Mixed Precision (AMP) training
        scaler = GradScaler(enabled=(config.ddp_backend == "nccl"))

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
            # Load scaler state if available (crucial for resuming AMP training)
            if "scaler_state_dict" in checkpoint:
                scaler.load_state_dict(checkpoint["scaler_state_dict"])
            start_epoch = checkpoint["epoch"] + 1

        # 3. TRAINING LOOP
        global_step = 0
        for epoch in range(start_epoch, EPOCHS):
            # Set model to training mode
            ddp_model.train()
            epoch_loss = 0.0
            for i, (inputs, targets) in enumerate(dataloader):
                inputs = inputs.to(local_rank)
                targets = targets.to(local_rank)

                # Mixed Precision Context
                with torch.amp.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                    enabled=(config.ddp_backend == "nccl")
                ):
                    outputs = ddp_model(inputs)
                    loss = criterion(outputs, targets)
                    # Scale the loss for gradient accumulation
                    loss = loss / GRAD_ACCUMULATION_STEPS

                epoch_loss += loss.item()

                # Backward pass with scaling
                scaler.scale(loss).backward()

                if (i + 1) % GRAD_ACCUMULATION_STEPS == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                global_step += 1
                if rank == 0 and global_step % LOG_STEP_INTERVAL == 0:
                    mlflow.log_metric("step_loss", loss.item() * GRAD_ACCUMULATION_STEPS, step=global_step)

            # Log epoch metrics from the master process only.
            if rank == 0:
                avg_epoch_loss = epoch_loss / max(1, len(dataloader))
                print(f"Epoch {epoch+1}, Avg Loss: {avg_epoch_loss:.4f}")
                mlflow.log_metric("loss", avg_epoch_loss, step=epoch)

                # Save checkpoint at the end of every epoch
                print(f"Saving checkpoint to {checkpoint_path}")
                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": ddp_model.module.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
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
                registered_model_name="nyc-taxi-regressor-ddp",
            )
            print(f"Model saved to {model_output_path}")
            mlflow.end_run()

    finally:
        # Ensure cleanup happens even if training crashes
        cleanup_ddp()


if __name__ == "__main__":
    run_training(
        data_path=config.processed_data_path,
        model_output_path=config.model_output_path,
    )
