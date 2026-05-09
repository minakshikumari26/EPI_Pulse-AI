from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator

from src.anomaly_detection.detect_spikes import detect_spikes
from src.forecasting.arima_model import arima_forecast
from src.preprocessing.feature_engineering import create_features
from src.preprocessing.clean_data import load_and_clean_data


def collect_data():
    df = load_and_clean_data()
    print(f"Collected {len(df)} outbreak rows")


def preprocess_outbreak_data():
    df = create_features()
    print(f"Generated features for {len(df)} rows")


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
    collect_data_task = PythonOperator(
        task_id="collect_data",
        python_callable=collect_data,
    )

    preprocess_data_task = PythonOperator(
        task_id="preprocess_data",
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

    collect_data_task >> preprocess_data_task >> detect_outbreaks_task >> train_forecast_task
