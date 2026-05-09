import pandas as pd
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "disease_data.csv"


def create_features():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df = df.sort_values(["region", "date"]).reset_index(drop=True)

    region_cases = df.groupby("region")["cases"]

    df["rolling_avg_cases"] = region_cases.transform(
        lambda cases: cases.rolling(window=3, min_periods=1).mean()
    )
    df["previous_day_cases"] = region_cases.shift(1)
    df["growth_rate"] = region_cases.pct_change().fillna(0)

    return df


if __name__ == "__main__":
    create_features()
