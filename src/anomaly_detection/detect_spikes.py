import pandas as pd
from scipy.stats import zscore
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "disease_data.csv"


def detect_spikes(threshold=1.5):
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])

    # Calculate z-score within each region so every city is compared to its own baseline.
    df["z_score"] = df.groupby("region")["cases"].transform(zscore)

    # Detect possible outbreak spikes.
    spikes = df[df["z_score"] > threshold].sort_values(["region", "date"])
    return spikes


if __name__ == "__main__":
    detect_spikes()
