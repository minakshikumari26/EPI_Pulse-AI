import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "disease_data.csv"


def arima_forecast(steps=7, order=(1, 1, 1)):
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    forecasts = []

    for region, region_df in df.sort_values("date").groupby("region"):
        series = region_df["cases"].reset_index(drop=True)
        model = ARIMA(series, order=order)
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=steps)

        future_dates = pd.date_range(
            start=region_df["date"].max() + pd.Timedelta(days=1),
            periods=steps,
            freq="D",
        )

        forecasts.append(
            pd.DataFrame(
                {
                    "region": region,
                    "date": future_dates,
                    "forecast_cases": forecast.round(2).to_numpy(),
                }
            )
        )

    forecast_df = pd.concat(forecasts, ignore_index=True)
    return forecast_df


if __name__ == "__main__":
    arima_forecast()
