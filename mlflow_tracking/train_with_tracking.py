import os

import mlflow
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.forecasting.arima_model import arima_forecast
from src.utils.config import load_config, project_path


def run_tracking_experiment():
    config = load_config()
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", str(project_path(config.get("mlflow", {}).get("tracking_uri", "mlruns"))))

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(config.get("mlflow", {}).get("experiment_name", "EpiPulse Forecasting"))

    forecast = arima_forecast(steps=7)
    y_true = forecast["forecast_cases"].to_numpy()
    y_pred = np.repeat(y_true.mean(), len(y_true))
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5

    with mlflow.start_run():
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_param("model", "ARIMA")
        mlflow.log_param("forecast_rows", len(forecast))

    print("Experiment logged")
    return {"mae": mae, "rmse": rmse}


if __name__ == "__main__":
    run_tracking_experiment()
