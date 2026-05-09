from fastapi import APIRouter
import pandas as pd

from api.schemas.prediction_schema import DiseaseInput, PredictionResponse
from src.alerting.alerts import generate_alert
from src.utils.config import get_data_path, load_config


router = APIRouter(tags=["prediction"])


@router.post("/predict", response_model=PredictionResponse)
def predict_outbreak(data: DiseaseInput):
    weights = load_config().get("risk_scoring", {}).get("weights", {})
    baseline_df = pd.read_csv(get_data_path())
    region_df = baseline_df[baseline_df["region"].str.lower() == data.region.lower()]

    if region_df.empty:
        region_df = baseline_df

    risk_score = round(
        _scale(data.cases, region_df["cases"]) * weights.get("cases", 60)
        + _scale(data.humidity, region_df["humidity"]) * weights.get("humidity", 25)
        + _scale(data.rainfall, region_df["rainfall"]) * weights.get("rainfall", 15),
        2,
    )
    alert = generate_alert(risk_score)

    return {
        "region": data.region,
        "risk_score": risk_score,
        "alert": alert,
    }


def _scale(value, baseline):
    value_range = baseline.max() - baseline.min()
    if value_range == 0:
        return 0
    return max(0, min(1, (value - baseline.min()) / value_range))
