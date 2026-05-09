import pandas as pd


def preprocess_data(path):
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.drop_duplicates()
    df = df.sort_values(["region", "date"]).reset_index(drop=True)

    numeric_cols = ["cases", "temperature", "humidity", "rainfall"]
    df[numeric_cols] = df.groupby("region")[numeric_cols].transform(lambda values: values.ffill().bfill())

    df["rolling_avg"] = df.groupby("region")["cases"].transform(
        lambda cases: cases.rolling(3, min_periods=1).mean()
    )
    df["growth_rate"] = df.groupby("region")["cases"].pct_change().fillna(0)

    return df

