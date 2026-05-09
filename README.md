EpiPulse AI
===========

EpiPulse AI is a prototype public-health analytics project for tracking disease case trends, detecting possible outbreak spikes, forecasting short-term case counts, and estimating regional risk.

Current Data
------------

The project uses `data/disease_data.csv`, a synthetic 90-day dataset across 8 Indian regions:

- Delhi
- Mumbai
- Bengaluru
- Chennai
- Kolkata
- Hyderabad
- Ahmedabad
- Pune

Each row contains:

- `date`
- `region`
- `cases`
- `temperature`
- `humidity`
- `rainfall`

Project Structure
-----------------

- `configs/config.yaml`: Central settings for data paths, thresholds, database, logging, forecasting, and geospatial coordinates.
- `api/main.py`: FastAPI service for cases, regions, risk, spikes, and simple forecasts.
- `api/routes/`: API route modules, including outbreak prediction.
- `api/schemas/`: Pydantic request and response schemas.
- `dashboard/app.py`: Streamlit dashboard with region filters and case trend charts.
- `docker/`: Separate Dockerfiles for the API and dashboard services.
- `database/db_connection.py`: SQLAlchemy database engine factory.
- `database/db.py`: Enterprise SQLAlchemy session and base setup.
- `database/models.py`: Database table models.
- `kafka/`: Producer and consumer helpers for disease event streaming.
- `airflow/`: Daily outbreak pipeline DAG.
- `mlflow_tracking/`: Example experiment tracking workflow.
- `k8s/`: Kubernetes deployment and service manifests.
- `monitoring/`: Monitoring/logging entrypoints.
- `src/preprocessing/clean_data.py`: Loads data, removes duplicates, fills missing numeric values, and parses dates.
- `src/preprocessing/feature_engineering.py`: Creates region-wise rolling average, lag, and growth-rate features.
- `src/alerting/alert_generator.py`: Creates outbreak alert labels from cases, risk score, z-score, and growth rate.
- `src/anomaly_detection/detect_spikes.py`: Detects region-wise z-score outbreak spikes.
- `src/anomaly_detection/isolation_forest_detector.py`: Detects multivariate anomalies per region.
- `src/forecasting/prophet_model.py`: Runs Prophet forecasts per region.
- `src/forecasting/arima_model.py`: Runs ARIMA forecasts per region.
- `src/deep_learning/lstm_model.py`: Provides LSTM sequence preparation and model-building helpers.
- `src/geospatial/heatmap.py`: Generates a regional outbreak heatmap.
- `src/risk_scoring/risk_score.py`: Calculates normalized regional risk scores.
- `src/utils/config.py`: Loads central configuration and resolves project paths.
- `src/utils/logger.py`: Creates reusable rotating file loggers.
- `src/visualization/plots.py`: Plots separate disease case trends by region.
- `notebooks/`: Jupyter lesson notebooks that explain how to run each project module and what output to expect.
- `notebooks/lesson_08_enterprise_layer.ipynb`: Guide to the Layer 4 enterprise architecture.

Setup
-----

```powershell
pip install -r requirements.txt
```

Optional enterprise dependencies:

```powershell
pip install -r requirements-enterprise.txt
```

Development/test dependencies:

```powershell
pip install -r requirements-dev.txt
```

Environment setup:

```powershell
copy .env.example .env
```

Run the dashboard:

```powershell
streamlit run dashboard/app.py
```

Run the API:

```powershell
uvicorn api.main:app --reload
```

LLM setup notes:

```powershell
ollama pull llama3.2:1b
ollama serve
```

If your machine has limited RAM, use a smaller local model such as `llama3.2:1b` and run Ollama in CPU-only mode if your version supports it.

```powershell
ollama serve --cpu
```

If `ollama serve --cpu` is not supported by your Ollama version, update Ollama or continue with the standard `ollama serve` command.

If `llama3.2:1b` still fails, use the models listed by `ollama list` and configure `configs/config.yaml` with one of the available local models.

You can also override LLM settings with environment variables:

```powershell
$env:OLLAMA_URL = "http://localhost:11435"
$env:LLM_MODEL = "llama3.2:1b"
$env:LLM_TIMEOUT = "60"
$env:LLM_TEMPERATURE = "0.7"
$env:LLM_ENABLED = "true"
```

By default the app supports fallback models via `llm.model_fallbacks` and will select the first available compatible model.

Run with Docker Compose:

```powershell
docker compose up --build
```

Enterprise layer commands:

```powershell
python kafka/producer.py
python kafka/consumer.py
python mlflow_tracking/train_with_tracking.py
airflow standalone
kubectl apply -f k8s/
```

Run individual modules:

```powershell
python src/preprocessing/clean_data.py
python src/preprocessing/feature_engineering.py
python src/anomaly_detection/detect_spikes.py
python src/anomaly_detection/isolation_forest_detector.py
python src/forecasting/prophet_model.py
python src/forecasting/arima_model.py
python src/risk_scoring/risk_score.py
python src/visualization/plots.py
python src/geospatial/heatmap.py
```

Run tests:

```powershell
python -m unittest discover -s tests
```

Useful API endpoints:

- `GET /`
- `GET /health`
- `GET /metrics`
- `GET /regions`
- `GET /cases`
- `GET /cases?region=Delhi`
- `GET /risk`
- `GET /spikes`
- `GET /forecast/simple`
- `POST /predict`
