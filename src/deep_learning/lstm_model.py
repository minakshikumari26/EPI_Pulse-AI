import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import get_data_path


def create_sequences(values, window_size=7):
    x_values = []
    y_values = []

    for index in range(len(values) - window_size):
        x_values.append(values[index : index + window_size])
        y_values.append(values[index + window_size])

    return np.array(x_values), np.array(y_values)


def build_lstm_model(window_size=7):
    from tensorflow.keras.layers import LSTM, Dense
    from tensorflow.keras.models import Sequential

    model = Sequential(
        [
            LSTM(50, activation="relu", input_shape=(window_size, 1)),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    return model


def prepare_region_data(region, window_size=7):
    df = pd.read_csv(get_data_path(), parse_dates=["date"])
    region_df = df[df["region"].str.lower() == region.lower()].sort_values("date")

    if region_df.empty:
        raise ValueError(f"No data found for region: {region}")

    values = region_df["cases"].astype(float).to_numpy()
    x_values, y_values = create_sequences(values, window_size=window_size)
    return x_values.reshape((x_values.shape[0], x_values.shape[1], 1)), y_values


def train_lstm(region="Delhi", window_size=7, epochs=20, batch_size=8):
    x_values, y_values = prepare_region_data(region, window_size=window_size)
    model = build_lstm_model(window_size=window_size)
    model.fit(x_values, y_values, epochs=epochs, batch_size=batch_size, verbose=0)
    return model


if __name__ == "__main__":
    model = build_lstm_model()
    model.summary()
