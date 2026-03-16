from typing import Optional

from pydantic_settings import BaseSettings
from pydantic_settings import SettingsConfigDict


class AppConfig(BaseSettings):
    # Model serving and checkpoint paths
    model_path: str = "models/model.pkl"
    model_output_path: str = "models/model.pt"

    # Dataset paths
    data_path: str = "data"
    processed_data_path: str = "data/processed"

    # DDP training config
    ddp_backend: str = "nccl"
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    batch_size: int = 4096
    num_workers: int = 4
    grad_accumulation_steps: int = 4
    epochs: int = 5
    log_step_interval: int = 100

    # Orchestration
    nproc_per_node: int = 2
    rdzv_endpoint: Optional[str] = None
    rdzv_id: str = "ddp_run"

    # Inference
    max_batch_size: int = 256

    # Tracking
    mlflow_tracking_uri: str = "http://localhost:5000"
    experiment_name: str = "nyc-taxi-fare-prediction"

    # API / UI
    api_url: str = "http://localhost:8000"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


config = AppConfig()
