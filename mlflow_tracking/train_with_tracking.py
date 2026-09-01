"""Real ARIMA vs Prophet benchmark, logged to MLflow.

For a representative region, holds out the last 30 days, trains each model on
the training window, forecasts the horizon, and logs MAE + RMSE against the
held-out actuals. Each model is one MLflow run so they show up side-by-side in
the MLflow UI.
"""

import os
import sys
import warnings
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import get_data_path, load_config, project_path


REGION = "Delhi"
HORIZON = 30
ARIMA_ORDER = (1, 1, 1)


def _load_region_series(region: str) -> pd.DataFrame:
    df = pd.read_csv(get_data_path(), parse_dates=["date"])
    region_df = df[df["region"] == region].copy()
    region_df = (
        region_df.groupby("date", as_index=False)["cases"]
        .sum()
        .sort_values("date")
        .reset_index(drop=True)
    )
    return region_df


def _train_test_split(series_df: pd.DataFrame, horizon: int):
    train = series_df.iloc[:-horizon]
    test = series_df.iloc[-horizon:]
    return train, test


def _score(y_true, y_pred) -> tuple[float, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(mean_squared_error(y_true, y_pred) ** 0.5)
    return mae, rmse


def _run_arima(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    model = ARIMA(train["cases"].to_numpy(), order=ARIMA_ORDER)
    fitted = model.fit()
    preds = np.asarray(fitted.forecast(steps=len(test)))
    mae, rmse = _score(test["cases"].to_numpy(), preds)
    return {"mae": mae, "rmse": rmse, "predictions": preds}


def _run_prophet(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from prophet import Prophet

        prophet_df = train.rename(columns={"date": "ds", "cases": "y"})
        model = Prophet()
        model.fit(prophet_df)

        future = model.make_future_dataframe(periods=len(test))
        forecast = model.predict(future).tail(len(test))

    preds = forecast["yhat"].to_numpy()
    mae, rmse = _score(test["cases"].to_numpy(), preds)
    return {"mae": mae, "rmse": rmse, "predictions": preds}


def run_tracking_experiment(region: str = REGION, horizon: int = HORIZON) -> dict:
    config = load_config()
    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        str(project_path(config.get("mlflow", {}).get("tracking_uri", "mlruns"))),
    )
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(config.get("mlflow", {}).get("experiment_name", "EpiPulse Forecasting"))

    series_df = _load_region_series(region)
    if len(series_df) <= horizon:
        raise ValueError(
            f"Not enough data for region {region!r}: need > {horizon} rows, got {len(series_df)}"
        )
    train, test = _train_test_split(series_df, horizon)

    results = {}

    with mlflow.start_run(run_name=f"arima_{region}"):
        mlflow.log_param("model", "ARIMA")
        mlflow.log_param("region", region)
        mlflow.log_param("forecast_horizon", horizon)
        mlflow.log_param("arima_order", str(ARIMA_ORDER))
        mlflow.log_param("train_rows", len(train))

        arima_result = _run_arima(train, test)
        mlflow.log_metric("mae", arima_result["mae"])
        mlflow.log_metric("rmse", arima_result["rmse"])
        results["arima"] = {"mae": arima_result["mae"], "rmse": arima_result["rmse"]}

    with mlflow.start_run(run_name=f"prophet_{region}"):
        mlflow.log_param("model", "Prophet")
        mlflow.log_param("region", region)
        mlflow.log_param("forecast_horizon", horizon)
        mlflow.log_param("train_rows", len(train))

        prophet_result = _run_prophet(train, test)
        mlflow.log_metric("mae", prophet_result["mae"])
        mlflow.log_metric("rmse", prophet_result["rmse"])
        results["prophet"] = {"mae": prophet_result["mae"], "rmse": prophet_result["rmse"]}

    print(f"Region: {region}, horizon: {horizon} days")
    for name, scores in results.items():
        print(f"  {name:8s}  MAE={scores['mae']:.2f}  RMSE={scores['rmse']:.2f}")
    return results


if __name__ == "__main__":
    run_tracking_experiment()
