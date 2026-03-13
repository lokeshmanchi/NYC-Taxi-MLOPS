"""Live Prediction page — call the FastAPI model endpoint."""

import os

import httpx
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Live Prediction — NYC Taxi", page_icon="🔮", layout="centered")
st.title("🔮 Live Fare Prediction")
st.markdown("Enter trip details below to get a predicted fare from the trained XGBoost model.")

# ── API health check ──────────────────────────────────────────────────────────
@st.cache_data(ttl=10)
def check_api(api_url: str):
    try:
        r = httpx.get(f"{api_url}/health", timeout=3)
        data = r.json()
        return data.get("status") == "ok", data.get("model_loaded", False)
    except Exception as e:
        return False, False


api_ok, model_loaded = check_api(API_URL)

if not api_ok:
    st.error(f"API is not reachable at `{API_URL}`. Make sure the API service is running.")
elif not model_loaded:
    st.warning("API is running but model is not loaded. Run `make train` first.")
else:
    st.success("API connected · Model loaded")

st.markdown("---")

# ── Input form ────────────────────────────────────────────────────────────────
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        trip_distance = st.number_input("Trip Distance (miles)", min_value=0.1, max_value=60.0, value=2.5, step=0.1)
        passenger_count = st.selectbox("Passengers", [1, 2, 3, 4, 5, 6], index=0)
        PULocationID = st.number_input("Pickup Zone ID (PULocationID)", min_value=1, max_value=265, value=236)
        DOLocationID = st.number_input("Dropoff Zone ID (DOLocationID)", min_value=1, max_value=265, value=237)

    with col2:
        pickup_hour = st.slider("Pickup Hour", 0, 23, 14)
        pickup_dayofweek = st.selectbox(
            "Day of Week",
            options=[0, 1, 2, 3, 4, 5, 6],
            format_func=lambda x: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][x],
            index=2,
        )
        pickup_month = st.selectbox("Month", [1, 2, 3, 4, 5], format_func=lambda x: ["Jan", "Feb", "Mar", "Apr", "May"][x - 1])
        RatecodeID = st.selectbox(
            "Rate Code",
            options=[1, 2, 3, 4, 5, 6],
            format_func=lambda x: {1: "Standard", 2: "JFK", 3: "Newark", 4: "Nassau/Westchester", 5: "Negotiated", 6: "Group ride"}[x],
        )

    submitted = st.form_submit_button("Predict Fare", use_container_width=True, type="primary")

# ── Prediction ────────────────────────────────────────────────────────────────
if submitted:
    payload = {
        "trip_distance": trip_distance,
        "passenger_count": passenger_count,
        "PULocationID": int(PULocationID),
        "DOLocationID": int(DOLocationID),
        "pickup_hour": pickup_hour,
        "pickup_dayofweek": pickup_dayofweek,
        "pickup_month": pickup_month,
        "RatecodeID": RatecodeID,
    }

    with st.spinner("Predicting…"):
        try:
            response = httpx.post(f"{API_URL}/predict", json=payload, timeout=5)
            if response.status_code == 200:
                result = response.json()
                st.markdown("---")
                st.markdown(
                    f"<div style='text-align:center; padding:2rem; background:#0e1117; border-radius:12px; border:1px solid #00CC96'>"
                    f"<h2 style='color:#00CC96; margin:0'>Predicted Fare</h2>"
                    f"<h1 style='color:#ffffff; font-size:3.5rem; margin:0.5rem 0'>${result['predicted_fare']:.2f}</h1>"
                    f"<p style='color:#888; margin:0'>USD · Base fare only (excluding tip, surcharges, tolls)</p>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                with st.expander("Request payload"):
                    st.json(payload)
            elif response.status_code == 503:
                st.error("Model not loaded on the server. Run training first.")
            else:
                st.error(f"API error {response.status_code}: {response.text}")
        except httpx.ConnectError:
            st.error(f"Cannot connect to API at `{API_URL}`.")
        except Exception as e:
            st.error(f"Unexpected error: {e}")

st.markdown("---")
st.caption(f"API endpoint: `{API_URL}/predict` · [Interactive API docs]({API_URL}/docs)")
