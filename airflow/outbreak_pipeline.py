"""Airflow DAG orchestrating the EpiPulse outbreak pipeline.

Threads the Kafka producer/consumer and Postgres ingest through the same run so
the streaming path and the analytical path share one source of truth.

Flow:
    seed_postgres → produce_events → consume_to_postgres
        → preprocess → detect_outbreaks → train_forecast_model
"""

import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from airflow import DAG
from airflow.operators.python import PythonOperator

from database.ingest import reload_disease_records_from_csv
from kafka.consumer import consume_and_persist
from kafka.producer import stream_disease_data
from src.anomaly_detection.detect_spikes import detect_spikes
from src.forecasting.arima_model import arima_forecast
from src.preprocessing.feature_engineering import create_features


STREAM_BATCH = 100


def seed_postgres():
    inserted = reload_disease_records_from_csv()
    print(f"Seeded {inserted} rows into disease_records")


def produce_events():
    events = stream_disease_data(limit=STREAM_BATCH)
    print(f"Produced {len(events)} disease events to Kafka")


def consume_to_postgres():
    inserted = consume_and_persist(max_messages=STREAM_BATCH)
    print(f"Consumed and persisted {inserted} events from Kafka")


def preprocess_outbreak_data():
    features = create_features()
    print(f"Generated features for {len(features)} rows")


def detect_outbreaks():
    spikes = detect_spikes()
    print(f"Detected {len(spikes)} spike rows")


def train_forecast_model():
    forecast = arima_forecast()
    print(f"Generated {len(forecast)} forecast rows")


with DAG(
    dag_id="epipulse_pipeline",
    start_date=datetime(2025, 1, 1),
    schedule="@daily",
    catchup=False,
) as dag:
    seed_postgres_task = PythonOperator(
        task_id="seed_postgres",
        python_callable=seed_postgres,
    )
    produce_events_task = PythonOperator(
        task_id="produce_events",
        python_callable=produce_events,
    )
    consume_to_postgres_task = PythonOperator(
        task_id="consume_to_postgres",
        python_callable=consume_to_postgres,
    )
    preprocess_task = PythonOperator(
        task_id="preprocess",
        python_callable=preprocess_outbreak_data,
    )
    detect_outbreaks_task = PythonOperator(
        task_id="detect_outbreaks",
        python_callable=detect_outbreaks,
    )
    train_forecast_task = PythonOperator(
        task_id="train_forecast_model",
        python_callable=train_forecast_model,
    )

    (
        seed_postgres_task
        >> produce_events_task
        >> consume_to_postgres_task
        >> preprocess_task
        >> detect_outbreaks_task
        >> train_forecast_task
    )
