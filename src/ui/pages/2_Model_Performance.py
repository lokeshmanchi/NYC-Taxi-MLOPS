"""Model Performance page — MLflow metrics and feature importance."""

import os

import mlflow
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = "nyc-taxi-fare-prediction"

st.set_page_config(page_title="Model Performance — NYC Taxi", page_icon="🤖", layout="wide")
st.title("🤖 Model Performance")

from src.features.transform import FEATURE_COLS


@st.cache_data(ttl=30)
def load_mlflow_runs(tracking_uri: str, experiment_name: str):
    mlflow.set_tracking_uri(tracking_uri)
    try:
        exp = mlflow.get_experiment_by_name(experiment_name)
        if exp is None:
            return None, None
        runs = mlflow.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=["start_time DESC"],
        )
        return exp, runs
    except Exception as e:
        return None, str(e)


@st.cache_data(ttl=30)
def load_model_feature_importance(tracking_uri: str, run_id: str):
    mlflow.set_tracking_uri(tracking_uri)
    try:
        # Load the pipeline model logged by the training script
        model_uri = f"runs:/{run_id}/model"
        pipeline = mlflow.sklearn.load_model(model_uri)
        
        # Extract the XGBoost model from the last step of the pipeline
        xgb_model = pipeline.named_steps['model']
        scores = xgb_model.feature_importances_
        feature_names = FEATURE_COLS
        fi = pd.DataFrame({"feature": feature_names, "importance": scores})
        fi = fi.sort_values("importance", ascending=True)
        return fi
    except Exception as e:
        print(f"Error loading feature importance: {e}")
        return None


exp, runs = load_mlflow_runs(MLFLOW_TRACKING_URI, EXPERIMENT_NAME)

if exp is None:
    st.warning("No MLflow experiment found. Run `make train` to train the model first.")
    st.stop()

if isinstance(runs, str):
    st.error(f"Could not connect to MLflow at `{MLFLOW_TRACKING_URI}`: {runs}")
    st.stop()

if runs.empty:
    st.warning("No training runs found. Run `make train` first.")
    st.stop()

latest = runs.iloc[0]
run_id = latest["run_id"]

# ── Latest run metrics ────────────────────────────────────────────────────────
st.subheader("Latest Run Metrics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("MAE", f"${latest.get('metrics.mae', float('nan')):.2f}")
col2.metric("RMSE", f"${latest.get('metrics.rmse', float('nan')):.2f}")
col3.metric("R²", f"{latest.get('metrics.r2', float('nan')):.4f}")
col4.metric("Run ID", run_id[:8] + "…")

# ── Feature importance ────────────────────────────────────────────────────────
st.subheader("Feature Importance")
fi = load_model_feature_importance(MLFLOW_TRACKING_URI, run_id)
if fi is not None:
    fig = px.bar(
        fi, x="importance", y="feature", orientation="h",
        color="importance", color_continuous_scale="viridis",
    )
    fig.update_layout(showlegend=False, height=400, xaxis_title="Importance Score")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Feature importance unavailable (model artifact not found in MLflow).")

# ── Run history table ─────────────────────────────────────────────────────────
st.subheader("All Training Runs")
display_cols = ["run_id", "start_time", "metrics.mae", "metrics.rmse", "metrics.r2",
                "params.n_estimators", "params.max_depth", "params.learning_rate"]
available = [c for c in display_cols if c in runs.columns]
runs_display = runs[available].copy()
runs_display.columns = [c.replace("metrics.", "").replace("params.", "") for c in available]
runs_display["run_id"] = runs_display["run_id"].str[:8]
runs_display["start_time"] = pd.to_datetime(runs_display["start_time"]).dt.strftime("%Y-%m-%d %H:%M")
st.dataframe(runs_display, use_container_width=True, hide_index=True)

# ── MAE over runs ─────────────────────────────────────────────────────────────
if "metrics.mae" in runs.columns and len(runs) > 1:
    st.subheader("MAE Across Runs")
    history = runs[["start_time", "metrics.mae", "metrics.rmse"]].dropna().sort_values("start_time")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=history["start_time"], y=history["metrics.mae"],
        mode="lines+markers", name="MAE",
    ))
    fig.add_trace(go.Scatter(
        x=history["start_time"], y=history["metrics.rmse"],
        mode="lines+markers", name="RMSE",
    ))
    fig.update_layout(xaxis_title="Run Time", yaxis_title="Error ($)", height=300)
    st.plotly_chart(fig, use_container_width=True)

st.caption(f"MLflow experiment: `{EXPERIMENT_NAME}` · Tracking URI: `{MLFLOW_TRACKING_URI}`")
