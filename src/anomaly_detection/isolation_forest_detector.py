import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "disease_data.csv"


def detect_anomalies(contamination=0.08):
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    feature_cols = ["cases", "temperature", "humidity", "rainfall"]
    detected_regions = []

    for region, region_df in df.groupby("region"):
        region_df = region_df.copy()
        scaled_features = StandardScaler().fit_transform(region_df[feature_cols])

        model = IsolationForest(
            contamination=contamination,
            random_state=42,
        )

        region_df["anomaly"] = model.fit_predict(scaled_features)
        region_df["anomaly_score"] = model.decision_function(scaled_features)
        detected_regions.append(region_df)

    result_df = pd.concat(detected_regions, ignore_index=True)
    anomalies = result_df[result_df["anomaly"] == -1].sort_values(["region", "date"])
    return anomalies


if __name__ == "__main__":
    detect_anomalies()
