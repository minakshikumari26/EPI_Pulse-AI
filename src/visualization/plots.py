import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "disease_data.csv"


def plot_cases(regions=None):
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df = df.sort_values(["region", "date"])

    if regions:
        df = df[df["region"].isin(regions)]

    fig, ax = plt.subplots(figsize=(12, 6))

    for region, region_df in df.groupby("region"):
        ax.plot(
            region_df["date"],
            region_df["cases"],
            marker="o",
            markersize=3,
            linewidth=2,
            label=region,
        )

    ax.set_title("Disease Case Trends by Region")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cases")
    ax.grid(True, alpha=0.3)
    ax.legend(title="Region", ncol=2)

    fig.autofmt_xdate()
    fig.tight_layout()
    plt.show()

    return fig


if __name__ == "__main__":
    plot_cases()
