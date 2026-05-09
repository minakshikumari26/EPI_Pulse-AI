import pandas as pd
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "disease_data.csv"


def _min_max_scale(series):
    value_range = series.max() - series.min()
    if value_range == 0:
        return series * 0
    return (series - series.min()) / value_range


def calculate_risk():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])

    for col in ["cases", "humidity", "rainfall"]:
        df[f"{col}_scaled"] = df.groupby("region")[col].transform(_min_max_scale)

    df["risk_score"] = (
        df["cases_scaled"] * 60 +
        df["humidity_scaled"] * 25 +
        df["rainfall_scaled"] * 15
    ).round(2)

    df["risk_level"] = df["risk_score"].apply(
        lambda score: "High" if score >= 70 else "Medium" if score >= 40 else "Low"
    )
    return df


if __name__ == "__main__":
    calculate_risk()
