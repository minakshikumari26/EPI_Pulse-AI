# EpiPulse AI

AI-powered disease outbreak intelligence and analysis.

EpiPulse AI is a full-stack MLOps project for tracking disease trends, detecting outbreak spikes, forecasting case counts, and generating LLM-powered public-health insights grounded in WHO-style guidance and IDSP protocols through RAG.

## What Is Included

| Feature | Details |
|---|---|
| RAG Chat | Ask questions in plain English and get guideline-aware answers with source citations. |
| Analytics Dashboard | Explore cross-regional risk, trends, spike detection, year-over-year comparisons, and 7-day projections. |
| Dynamic Data | Use the built-in dataset or upload a CSV with `date`, `region`, and `cases`. |
| Forecasting | ARIMA/Prophet modules for time-series disease case forecasting. |
| Anomaly Detection | Z-score and Isolation Forest based outbreak signal detection. |
| Risk Scoring | Weighted risk score based on cases, humidity, and rainfall. |
| API Backend | FastAPI endpoints for cases, risk, spikes, forecasts, and predictions. |
| MLOps Layer | Docker, Kubernetes manifests, Kafka, Airflow, MLflow, and monitoring scaffolding. |

## Dashboard Pages

The Streamlit app is organized as a multipage dashboard:

```text
dashboard/
  Home.py                 # Multipage entrypoint and landing page
  chat.py                 # Main RAG chat implementation
  app.py                  # Main analytics implementation
  Pages/
    1_chat.py             # Wrapper that runs chat.main()
    2_Analytics.py        # Wrapper that runs app.main()
```

The wrapper files keep the standalone pages reusable while still allowing one combined multipage app.

## Dataset

`data/disease_data.csv` is a generated synthetic dataset with realistic outbreak patterns.

It includes:

- 20 Indian states/cities, including Delhi, Mumbai, Bengaluru, Chennai, Kolkata, Hyderabad, Pune, Jaipur, and others.
- 5 diseases: Dengue, Malaria, Cholera, Typhoid, and COVID-19.
- Daily data across 2020-2025.
- Weather and seasonal features: temperature, humidity, and rainfall.
- Outbreak-like spikes and regional risk variation.

Expected columns:

```text
date, region, disease, cases, temperature, humidity, rainfall, year, month
```

## Project Structure

```text
EPI_Pulse_AI/
  api/
    main.py                         # FastAPI app entrypoint
    routes/
      prediction_routes.py          # Prediction and data API routes
    schemas/
      prediction_schema.py          # Pydantic request/response models
  airflow/
    outbreak_pipeline.py            # Daily outbreak pipeline DAG
  configs/
    config.yaml                     # Central app, model, data, and service config
  dashboard/
    Home.py                         # Streamlit multipage entrypoint
    app.py                          # Analytics dashboard implementation
    chat.py                         # RAG chat implementation
    Pages/
      1_chat.py                     # Streamlit page wrapper for chat.py
      2_Analytics.py                # Streamlit page wrapper for app.py
  data/
    disease_data.csv                # Main synthetic disease dataset
  database/
    db.py                           # SQLAlchemy session and base setup
    db_connection.py                # Database engine factory
    models.py                       # ORM table models
  docker/
    Dockerfile.api                  # API container image
    Dockerfile.dashboard            # Dashboard container image
  k8s/
    api-deployment.yaml             # API Kubernetes deployment
    dashboard-deployment.yaml       # Dashboard Kubernetes deployment
    postgres-deployment.yaml        # PostgreSQL Kubernetes deployment
  kafka/
    producer.py                     # Disease event producer
    consumer.py                     # Disease event consumer
  mlflow_tracking/
    train_with_tracking.py          # MLflow training/experiment script
  monitoring/
    logger.py                       # Monitoring logger setup
  notebooks/
    lesson_01_project_overview.ipynb
    lesson_02_data_and_preprocessing.ipynb
    lesson_03_feature_engineering_and_risk.ipynb
    lesson_04_anomaly_detection.ipynb
    lesson_05_forecasting.ipynb
    lesson_06_dashboard_and_api.ipynb
    lesson_07_alerting_geospatial_and_docker.ipynb
    lesson_08_enterprise_layer.ipynb
  src/
    alerting/
      alert_generator.py            # LLM-powered alert explanations
      alerts.py                     # Alert helpers
    anomaly_detection/
      detect_spikes.py              # Z-score spike detection
      isolation_forest_detector.py  # Isolation Forest anomaly detection
    deep_learning/
      lstm_model.py                 # LSTM forecasting model
    forecasting/
      arima_model.py                # ARIMA forecasting
      predict_cases.py              # Forecast runner/helpers
      prophet_model.py              # Prophet forecasting
    geospatial/
      heatmap.py                    # Folium outbreak heatmap
    llm/
      llm_client.py                 # Unified Groq/Ollama client
      ollama_client.py              # Ollama-specific client
    preprocessing/
      clean_data.py                 # Data cleaning
      feature_engineering.py        # Feature engineering
      preprocess.py                 # Preprocessing workflow
    rag/
      knowledge_base.py             # ChromaDB knowledge base
      retriever.py                  # RAG retrieval and answer builder
    risk_scoring/
      risk_score.py                 # Weighted outbreak risk scoring
    utils/
      config.py                     # Config loader
      logger.py                     # App logger
    visualization/
      plots.py                      # Plotly visualization helpers
  tests/
    test_api_routes.py              # API tests
    test_core_workflows.py          # Data/model workflow tests
    test_llm_client.py              # LLM client tests
  docker-compose.yml                # Local multi-service setup
  generate_dataset.py               # Dataset generator
  requirements.txt                  # Core Python dependencies
  requirements-dev.txt              # Development/test dependencies
  requirements-enterprise.txt       # Optional enterprise dependencies
```

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Configure LLM access:

```bash
copy .env.example .env
```

Then set the required values in `.env`, for example:

```text
GROQ_API_KEY=gsk_your_key_here
LLM_PROVIDER=groq
```

Run the multipage dashboard:

```bash
streamlit run dashboard/Home.py
```

Run pages individually:

```bash
streamlit run dashboard/chat.py
streamlit run dashboard/app.py
```

Run the API:

```bash
uvicorn api.main:app --reload
```

## CSV Upload Format

The dashboard accepts CSV uploads with at least these columns:

```text
date, region, cases
```

Optional columns:

```text
disease, temperature, humidity, rainfall
```

## LLM Configuration

The main settings live in `configs/config.yaml`.

Groq cloud:

```yaml
llm:
  enabled: true
  provider: groq
  model: llama-3.1-8b-instant
```

Ollama local:

```yaml
llm:
  enabled: true
  provider: ollama
  ollama_url: http://localhost:11434
  ollama_model: llama3.2:1b
```

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check |
| GET | `/health` | Service status |
| GET | `/metrics` | Prometheus metrics |
| GET | `/regions` | List regions |
| GET | `/cases` | List cases |
| GET | `/cases?region=Delhi` | Filter cases by region |
| GET | `/risk` | Risk scores |
| GET | `/spikes` | Detected spikes |
| GET | `/forecast/simple` | Simple 7-day forecast |
| POST | `/predict` | ARIMA prediction |

## Tests

```bash
python -m unittest discover -s tests
python test_rag.py
```

## Docker

```bash
docker compose up --build
```

Services:

- Dashboard: Streamlit on port `8501`
- API: FastAPI on port `8000`
- Database: PostgreSQL on port `5432`

## License

MIT
