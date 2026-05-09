import sys
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, Query, Response

try:
    from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, generate_latest
except ModuleNotFoundError:
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4"
    Counter = None
    Gauge = None

    def generate_latest():
        return b"# epipulse metrics unavailable: prometheus_client is not installed\n"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import get_data_path, load_config
from api.routes.prediction_routes import router as prediction_router
from src.alerting.alert_generator import generate_alert_with_explanation


config = load_config()
app = FastAPI(title=config.get("app", {}).get("name", "EpiPulse AI"))
app.include_router(prediction_router)

REQUEST_COUNTER = (
    Counter(
        "epipulse_api_requests_total",
        "Total API requests served by EpiPulse AI",
        ["endpoint"],
    )
    if Counter
    else None
)
REGION_GAUGE = (
    Gauge(
        "epipulse_regions_total",
        "Number of regions available in the disease dataset",
    )
    if Gauge
    else None
)
SPIKE_GAUGE = (
    Gauge(
        "epipulse_spike_days_total",
        "Number of currently detected spike rows",
    )
    if Gauge
    else None
)


def _count_request(endpoint):
    if REQUEST_COUNTER is not None:
        REQUEST_COUNTER.labels(endpoint=endpoint).inc()


def _load_data():
    df = pd.read_csv(get_data_path(), parse_dates=["date"])
    return df.sort_values(["region", "date"]).reset_index(drop=True)


def _filter_cases(df, region: Optional[str], start_date: Optional[str], end_date: Optional[str]):
    if region:
        df = df[df["region"].str.lower() == region.lower()]
    if start_date:
        df = df[df["date"] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df["date"] <= pd.to_datetime(end_date)]
    return df


def _risk_scored_data(df):
    weights = config.get("risk_scoring", {}).get("weights", {})
    levels = config.get("risk_scoring", {}).get("levels", {})

    for col in ["cases", "humidity", "rainfall"]:
        value_range = df.groupby("region")[col].transform(lambda values: values.max() - values.min())
        min_value = df.groupby("region")[col].transform("min")
        df[f"{col}_scaled"] = ((df[col] - min_value) / value_range).fillna(0)

    df["risk_score"] = (
        df["cases_scaled"] * weights.get("cases", 60)
        + df["humidity_scaled"] * weights.get("humidity", 25)
        + df["rainfall_scaled"] * weights.get("rainfall", 15)
    ).round(2)
    df["risk_level"] = df["risk_score"].apply(
        lambda score: "High"
        if score >= levels.get("high", 70)
        else "Medium"
        if score >= levels.get("medium", 40)
        else "Low"
    )
    return df


def _records(df):
    output = df.copy()
    output["date"] = output["date"].dt.strftime("%Y-%m-%d")
    return output.to_dict(orient="records")


@app.get("/")
def home():
    _count_request("/")
    return {"message": "EpiPulse AI API Running"}


@app.get("/health")
def health_check():
    _count_request("/health")
    data_path = get_data_path()
    return {
        "status": "healthy" if data_path.exists() else "degraded",
        "data_path": str(data_path),
        "data_available": data_path.exists(),
        "environment": config.get("app", {}).get("environment", "development"),
    }


@app.get("/metrics")
def metrics():
    df = _load_data()
    if REGION_GAUGE is not None:
        REGION_GAUGE.set(df["region"].nunique())
    threshold = config.get("anomaly_detection", {}).get("z_score_threshold", 1.5)
    z_scores = df.groupby("region")["cases"].transform(
        lambda values: (values - values.mean()) / values.std(ddof=0)
    )
    if SPIKE_GAUGE is not None:
        SPIKE_GAUGE.set(int((z_scores > threshold).sum()))
    _count_request("/metrics")
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/regions")
def get_regions():
    _count_request("/regions")
    df = _load_data()
    return {"regions": sorted(df["region"].unique())}


@app.get("/cases")
def get_cases(
    region: Optional[str] = None,
    start_date: Optional[str] = Query(default=None, description="YYYY-MM-DD"),
    end_date: Optional[str] = Query(default=None, description="YYYY-MM-DD"),
):
    _count_request("/cases")
    df = _filter_cases(_load_data(), region, start_date, end_date)
    return _records(df)


@app.get("/risk")
def get_risk(region: Optional[str] = None):
    _count_request("/risk")
    df = _risk_scored_data(_load_data())
    latest = df.sort_values("date").groupby("region").tail(1)
    if region:
        latest = latest[latest["region"].str.lower() == region.lower()]
    return _records(latest[["date", "region", "cases", "risk_score", "risk_level"]])


@app.get("/spikes")
def get_spikes(region: Optional[str] = None):
    _count_request("/spikes")
    df = _load_data()
    threshold = config.get("anomaly_detection", {}).get("z_score_threshold", 1.5)
    df["z_score"] = df.groupby("region")["cases"].transform(
        lambda values: (values - values.mean()) / values.std(ddof=0)
    )
    spikes = df[df["z_score"] > threshold]
    if region:
        spikes = spikes[spikes["region"].str.lower() == region.lower()]
    return _records(spikes[["date", "region", "cases", "z_score"]])


@app.get("/forecast/simple")
def get_simple_forecast(region: Optional[str] = None):
    _count_request("/forecast/simple")
    df = _load_data()
    if region:
        df = df[df["region"].str.lower() == region.lower()]

    periods = config.get("forecasting", {}).get("periods", 7)
    forecasts = []

    for region_name, region_df in df.groupby("region"):
        region_df = region_df.sort_values("date")
        last_date = region_df["date"].max()
        last_cases = float(region_df.iloc[-1]["cases"])
        recent_avg = region_df.tail(7)["cases"].mean()
        previous_avg = region_df.tail(14).head(7)["cases"].mean()
        daily_change = 0 if pd.isna(previous_avg) else (recent_avg - previous_avg) / 7

        for day in range(1, periods + 1):
            forecasts.append(
                {
                    "region": region_name,
                    "date": (last_date + pd.Timedelta(days=day)).strftime("%Y-%m-%d"),
                    "projected_cases": max(0, round(last_cases + daily_change * day, 1)),
                }
            )

    return forecasts


# LLM-Powered Endpoints
@app.get("/llm/explain-alert")
def explain_alert(
    region: str,
    cases: Optional[float] = None,
    risk_score: Optional[float] = None,
    z_score: Optional[float] = None,
    growth_rate: Optional[float] = None,
):
    """
    Get LLM-powered explanation for an outbreak alert.

    Args:
        region: Region name
        cases: Number of cases
        risk_score: Risk score (0-100)
        z_score: Z-score for anomaly detection
        growth_rate: Growth rate of cases

    Returns:
        Alert with LLM explanation
    """
    _count_request("/llm/explain-alert")
    try:
        result = generate_alert_with_explanation(
            cases=cases,
            risk_score=risk_score,
            z_score=z_score,
            growth_rate=growth_rate,
            region=region,
            use_llm=True,
        )
        return result
    except Exception as e:
        return {
            "error": str(e),
            "alert": "Unable to generate alert with explanation",
        }


@app.get("/llm/question")
def ask_llm(question: str, region: Optional[str] = None):
    """
    Ask the AI epidemiologist a question about the outbreak data.

    Args:
        question: Question about disease outbreaks
        region: Optional region for context

    Returns:
        LLM-generated answer
    """
    _count_request("/llm/question")
    try:
        config_data = load_config().get("llm", {})
        if not config_data.get("enabled", False):
            return {
                "error": "LLM is disabled",
                "message": "Enable LLM in config.yaml",
            }

        from src.llm import get_llm_client

        llm = get_llm_client()

        context = {"available_regions": sorted(_load_data()["region"].unique())}
        if region:
            context["selected_region"] = region

        answer = llm.answer_question(question, context=context)
        return {"question": question, "answer": answer}

    except ConnectionError:
        return {
            "error": "Cannot connect to Ollama",
            "message": (
                "Ensure Ollama is running: "
                "`ollama serve` and `ollama pull mistral`"
            ),
        }
    except Exception as e:
        return {
            "error": str(e),
            "message": "Error generating answer",
        }


@app.get("/llm/summarize")
def summarize_situation(region: Optional[str] = None):
    """
    Generate LLM-powered summary of current outbreak situation.

    Args:
        region: Optional specific region to summarize

    Returns:
        Narrative summary of the situation
    """
    _count_request("/llm/summarize")
    try:
        config_data = load_config().get("llm", {})
        if not config_data.get("enabled", False):
            return {
                "error": "LLM is disabled",
                "message": "Enable LLM in config.yaml",
            }

        from src.llm import get_llm_client

        llm = get_llm_client()
        df = _risk_scored_data(_load_data())
        latest = df.sort_values("date").groupby("region").tail(1)

        if region:
            latest = latest[latest["region"].str.lower() == region.lower()]

        summary_data = {
            "regions_analyzed": list(latest["region"]),
            "total_cases": int(latest["cases"].sum()),
            "avg_risk_score": float(latest["risk_score"].mean()),
            "high_risk_regions": list(
                latest[latest["risk_level"] == "High"]["region"]
            ),
        }

        summary = llm.generate_report(summary_data)
        return {"summary": summary}

    except ConnectionError:
        return {
            "error": "Cannot connect to Ollama",
            "message": (
                "Ensure Ollama is running: "
                "`ollama serve` and `ollama pull mistral`"
            ),
        }
    except Exception as e:
        return {
            "error": str(e),
            "message": "Error generating summary",
        }
