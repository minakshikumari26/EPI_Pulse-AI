import pandas as pd
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "disease_data.csv"


def load_and_clean_data():
    df = pd.read_csv(DATA_PATH)

    df = df.drop_duplicates()

    numeric_cols = ["cases", "temperature", "humidity", "rainfall"]
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].mean())

    df["date"] = pd.to_datetime(df["date"])

    return df


if __name__ == "__main__":
    load_and_clean_data()
