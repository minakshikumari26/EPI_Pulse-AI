# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

**EpiPulse AI** — a Streamlit + FastAPI application for disease outbreak analytics with a RAG-powered LLM chat, plus a Kafka + Airflow + Postgres pipeline that ingests synthetic disease events. The primary dataset is a **generated CSV** (`data/disease_data.csv`); the API and dashboard read directly from that CSV. Postgres is populated by the Airflow pipeline but not on the read path.

## Common commands

```bash
# Install (choose one tier)
pip install -r requirements.txt              # core: dashboard, API, RAG, LLM, tests
pip install -r requirements-enterprise.txt   # adds mlflow, airflow, kafka-python, redis, geopandas, tensorflow

# Regenerate the synthetic dataset (overwrites data/disease_data.csv)
python generate_dataset.py

# Streamlit (canonical entrypoint)
streamlit run dashboard/Home.py
# Or run individual pages directly:
streamlit run dashboard/chat.py
streamlit run dashboard/app.py

# FastAPI backend
uvicorn api.main:app --reload

# Tests
python -m unittest discover -s tests          # all unit tests
python -m unittest tests.test_api_routes      # a single test module
python -m unittest tests.test_api_routes.ApiRouteTests.test_health   # a single test
python test_rag.py                            # sanity-check RAG (seeds ChromaDB, runs a retrieval)

# MLflow experiment (ARIMA vs Prophet, 30-day held-out window for Delhi)
python mlflow_tracking/train_with_tracking.py
mlflow ui                                     # inspect runs

# Airflow DAG (requires local Postgres on 5432 and Kafka on 9094)
airflow standalone
# Then trigger dag_id=epipulse_pipeline in the UI
```

Tests import functions directly from `api.main` and call them in-process — they do **not** spin up an HTTP server or a database. `data/disease_data.csv` must exist.

## Environment / config

- `.env` (copy from `.env.example`) sets `LLM_PROVIDER`, `GROQ_API_KEY`, `OLLAMA_URL`, `OLLAMA_MODEL`, `DATABASE_URL`, `KAFKA_BOOTSTRAP_SERVERS`, `MLFLOW_TRACKING_URI`.
- `configs/config.yaml` holds thresholds (anomaly z-score, risk weights, alerting cutoffs), model choices, region coordinates.
- `src/utils/config.py` merges: `DEFAULT_CONFIG` (baked into the file) ← `configs/config.yaml` ← env vars. When adding a new setting, add it to `DEFAULT_CONFIG` too, or `load_config()` won't return it.
- **`configs/config.yaml` and `DEFAULT_CONFIG` can drift** — e.g., yaml uses `ollama_url: http://localhost:11434` while `DEFAULT_CONFIG` has `11435`, and yaml uses `model: llama-3.1-8b-instant` (Groq) while `DEFAULT_CONFIG` has `model: llama3.2:1b` (Ollama). Prefer yaml as the source of truth.

## Architecture — what to know before editing

**Path bootstrapping.** Modules under `src/`, `api/`, `database/`, `kafka/`, `airflow/`, and top-level scripts each insert `PROJECT_ROOT` into `sys.path` at import time. This lets any file run standalone. When adding a new entrypoint, follow the same pattern.

**Streamlit multipage layout.** `dashboard/Home.py` is the multipage root. `dashboard/chat.py` and `dashboard/app.py` are full standalone pages, each exposing a `main()` function. `dashboard/pages/1_chat.py` and `2_Analytics.py` are **thin wrappers** that just call `chat.main()` / `app.main()`. Preserve this split — don't inline chat/analytics logic into the wrapper files.

**LLM abstraction.** `src/llm/llm_client.py` defines `BaseLLMClient` with `generate()`, `answer_question()`, `explain_alert()`, `generate_report()`. Concrete providers (Groq, Ollama) subclass it. `get_llm_client()` picks the provider from config/env. `configs/config.yaml → llm.model_fallbacks` is a list of alternate model names to try if the primary fails.

**RAG pipeline.** `src/rag/knowledge_base.py` wraps ChromaDB + sentence-transformers with WHO/IDSP seed content. `src/rag/retriever.py` combines semantic retrieval with live disease-data context, applies `RELEVANCE_THRESHOLD = 0.20` and caps context at `MAX_CONTEXT_CHARS = 3000`. `protobuf==3.20.3` is pinned in `requirements.txt` because newer versions break the Streamlit ↔ ChromaDB ↔ sentence-transformers import chain.

**Airflow pipeline.** `airflow/outbreak_pipeline.py` defines DAG `epipulse_pipeline` with 6 tasks:
`seed_postgres → produce_events → consume_to_postgres → preprocess → detect_outbreaks → train_forecast_model`.
- `seed_postgres` — `database.ingest.reload_disease_records_from_csv()` truncates and full-reloads `disease_records`.
- `produce_events` — `kafka.producer.stream_disease_data(limit=100)` replays 100 rows from the CSV onto the Kafka topic.
- `consume_to_postgres` — `kafka.consumer.consume_and_persist(max_messages=100)` — bounded read (uses `consumer_timeout_ms=10000` so the task exits instead of hanging).
- The remaining tasks call existing preprocessing / anomaly / forecasting functions.

**MLflow experiment.** `mlflow_tracking/train_with_tracking.py` holds out the last 30 days for one region (`Delhi`), trains ARIMA and Prophet on the remainder, and logs MAE/RMSE per model as two separate MLflow runs (`arima_Delhi`, `prophet_Delhi`).

**Data flow (read path).** CSV → `pd.read_csv` inside `api/main.py._load_data()` and dashboard modules. Risk scoring, spike detection, and forecasts are computed on-the-fly per request from the CSV. Postgres is written to by the Airflow DAG but not read by the API or dashboard.

**API.** `api/main.py` mounts `api/routes/prediction_routes.py` (POST `/predict` for ARIMA). Prometheus counters/gauges are optional — if `prometheus_client` isn't installed, `/metrics` returns a stub. Tests call the route functions directly, so keep them importable.

## Gotchas

- The local `kafka/` folder can shadow the PyPI `kafka-python` package if imported from certain paths. `kafka/producer.py` and `kafka/consumer.py` import `from kafka import KafkaProducer/KafkaConsumer` inside functions and handle the `ImportError` — preserve that pattern.
- The synthetic dataset has ~20 regions (see `generate_dataset.py`), but `configs/config.yaml → geospatial.region_coordinates` only maps 8 of them. Regions without coordinates won't render on the Folium heatmap.
- `test_rag.py` triggers a ~80MB embedding-model download the first time it runs.
- The MLflow experiment uses Prophet, which has cmdstan / stan-related warnings on first import — suppressed in `mlflow_tracking/train_with_tracking.py`.
- `Base.metadata.create_all(engine, tables=[DiseaseRecord.__table__])` is called by both `database/ingest.py` and `kafka/consumer.py:consume_and_persist()` so either can be run first without a manual schema step.
