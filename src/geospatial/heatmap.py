import sys
from pathlib import Path

import pandas as pd
import folium
from folium.plugins import HeatMap

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import get_data_path, load_config, project_path


def create_heatmap(output_path="models/outbreak_heatmap.html", value_col="cases"):
    config = load_config().get("geospatial", {})
    coordinates = config.get("region_coordinates", {})

    df = pd.read_csv(get_data_path(), parse_dates=["date"])
    latest_df = df.sort_values("date").groupby("region").tail(1)

    heat_data = []
    for _, row in latest_df.iterrows():
        region_coordinates = coordinates.get(row["region"])
        if not region_coordinates:
            continue
        lat, lon = region_coordinates
        heat_data.append([lat, lon, float(row[value_col])])

    map_center = config.get("map_center", [22.9734, 78.6569])
    outbreak_map = folium.Map(
        location=map_center,
        zoom_start=config.get("map_zoom", 5),
        tiles="CartoDB positron",
    )

    if heat_data:
        HeatMap(heat_data, radius=28, blur=18).add_to(outbreak_map)

    for _, row in latest_df.iterrows():
        region_coordinates = coordinates.get(row["region"])
        if not region_coordinates:
            continue

        folium.CircleMarker(
            location=region_coordinates,
            radius=7,
            tooltip=f"{row['region']}: {row[value_col]} cases",
            color="#dc2626",
            fill=True,
            fill_opacity=0.75,
        ).add_to(outbreak_map)

    output_file = project_path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    outbreak_map.save(output_file)
    return outbreak_map


if __name__ == "__main__":
    create_heatmap()
