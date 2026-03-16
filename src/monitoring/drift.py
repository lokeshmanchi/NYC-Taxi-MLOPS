"""Drift detection: compare live prediction inputs against training distribution."""

import os

import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

from src.features.core import BASE_FEATURE_COLS
from src.config import config

DRIFT_REPORT_PATH = "data/drift_report.html"
PREDICTION_LOG_PATH = "data/predictions"


def generate_drift_report(
    reference_path: str = None,
    current_path: str = PREDICTION_LOG_PATH,
    output_path: str = DRIFT_REPORT_PATH,
) -> str:
    """
    Compare the distribution of live prediction inputs (current) against
    the processed training data (reference) and save an Evidently HTML report.

    Args:
        reference_path: Path to processed training Parquet (Hive-partitioned dir).
        current_path:   Path to prediction log Parquet (date-partitioned dir).
        output_path:    Where to write the HTML report.

    Returns:
        Absolute path to the generated HTML report.
    """
    if reference_path is None:
        reference_path = config.processed_data_path

    # Load reference — sample up to 5 000 rows to keep report generation fast.
    ref_df = pd.read_parquet(reference_path, columns=BASE_FEATURE_COLS)
    ref_df = ref_df.sample(n=min(5_000, len(ref_df)), random_state=42)

    # Load current prediction logs.
    if not os.path.exists(current_path):
        raise FileNotFoundError(
            f"No prediction logs found at '{current_path}'. "
            "Call /predict at least once before generating a drift report."
        )
    cur_df = pd.read_parquet(current_path, columns=BASE_FEATURE_COLS)

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref_df, current_data=cur_df)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    report.save_html(output_path)
    return os.path.abspath(output_path)
