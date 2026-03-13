FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir pandas pyarrow numpy scikit-learn xgboost joblib mlflow fastapi "uvicorn[standard]" pydantic python-dotenv

COPY src/ src/

CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
