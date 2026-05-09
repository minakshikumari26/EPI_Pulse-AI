import logging
import os
import tempfile
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "epipulse_matplotlib"))

for logger_name in ["cmdstanpy", "prophet"]:
    logging.getLogger(logger_name).setLevel(logging.ERROR)
    logging.getLogger(logger_name).propagate = False

from prophet import Prophet

DATA_PATH = PROJECT_ROOT / "data" / "disease_data.csv"


def forecast_cases(periods=7):
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    forecasts = []

    for region, region_df in df.groupby("region"):
        prophet_df = region_df[["date", "cases"]].rename(
            columns={
                "date": "ds",
                "cases": "y",
            }
        )

        model = Prophet()
        model.fit(prophet_df)

        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)
        forecast["region"] = region
        forecasts.append(forecast[["region", "ds", "yhat"]].tail(periods))

    return pd.concat(forecasts, ignore_index=True)


if __name__ == "__main__":
    print(forecast_cases())
