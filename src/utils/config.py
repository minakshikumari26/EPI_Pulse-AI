import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    load_dotenv = None

try:
    import yaml
except ModuleNotFoundError:
    yaml = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
ENV_PATH = PROJECT_ROOT / ".env"

if load_dotenv is not None:
    load_dotenv(ENV_PATH)

DEFAULT_CONFIG = {
    "app": {"name": "EpiPulse AI", "environment": "development"},
    "data": {"disease_data_path": "data/disease_data.csv"},
    "api": {"host": "0.0.0.0", "port": 8000},
    "dashboard": {"port": 8501},
    "database": {"url": "postgresql://postgres:password@db:5432/epipulse"},
    "logging": {
        "level": "INFO",
        "file_path": "logs/app.log",
        "max_bytes": 1048576,
        "backup_count": 3,
    },
    "anomaly_detection": {
        "z_score_threshold": 1.5,
        "isolation_forest_contamination": 0.08,
    },
    "risk_scoring": {
        "weights": {"cases": 60, "humidity": 25, "rainfall": 15},
        "levels": {"high": 70, "medium": 40},
    },
    "forecasting": {"periods": 7, "arima_order": [1, 1, 1]},
    "alerting": {
        "high_risk_score": 70,
        "medium_risk_score": 40,
        "spike_z_score": 1.5,
        "growth_rate_warning": 0.2,
    },
    "kafka": {
        "bootstrap_servers": "localhost:9094",
        "topic": "disease-events",
    },
    "mlflow": {
        "experiment_name": "EpiPulse Forecasting",
    },
    "llm": {
        "enabled": False,
        "ollama_url": "http://localhost:11435",
        "model": "llama3.2:1b",
        "model_fallbacks": ["llama3.2:1b", "mistral:latest", "mistral"],
        "timeout": 60,
        "temperature": 0.7,
    },
    "geospatial": {
        "map_center": [22.9734, 78.6569],
        "map_zoom": 5,
        "region_coordinates": {
            "Delhi": [28.6139, 77.2090],
            "Mumbai": [19.0760, 72.8777],
            "Bengaluru": [12.9716, 77.5946],
            "Chennai": [13.0827, 80.2707],
            "Kolkata": [22.5726, 88.3639],
            "Hyderabad": [17.3850, 78.4867],
            "Ahmedabad": [23.0225, 72.5714],
            "Pune": [18.5204, 73.8567],
        },
    },
}


def _deep_merge(defaults, overrides):
    merged = defaults.copy()
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config():
    if yaml is None or not CONFIG_PATH.exists():
        return _apply_env_overrides(DEFAULT_CONFIG)

    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    return _apply_env_overrides(_deep_merge(DEFAULT_CONFIG, config))


def _apply_env_overrides(config):
    updated = _deep_merge(DEFAULT_CONFIG, config)

    updated["app"]["environment"] = os.getenv("APP_ENV", updated["app"]["environment"])
    updated["database"]["url"] = os.getenv("DATABASE_URL", updated["database"]["url"])
    updated["kafka"]["bootstrap_servers"] = os.getenv(
        "KAFKA_BOOTSTRAP_SERVERS",
        updated["kafka"]["bootstrap_servers"],
    )
    updated["kafka"]["topic"] = os.getenv("KAFKA_TOPIC", updated["kafka"]["topic"])
    updated["llm"]["enabled"] = os.getenv(
        "LLM_ENABLED",
        str(updated["llm"].get("enabled", False)),
    ).lower() in {"1", "true", "yes", "on"}
    updated["llm"]["ollama_url"] = os.getenv(
        "OLLAMA_URL",
        updated["llm"]["ollama_url"],
    )
    updated["llm"]["model"] = os.getenv(
        "LLM_MODEL",
        updated["llm"]["model"],
    )
    updated["llm"]["timeout"] = int(
        os.getenv("LLM_TIMEOUT", updated["llm"]["timeout"])
    )
    updated["llm"]["temperature"] = float(
        os.getenv("LLM_TEMPERATURE", updated["llm"]["temperature"])
    )
    updated["mlflow"]["experiment_name"] = os.getenv(
        "MLFLOW_EXPERIMENT_NAME",
        updated["mlflow"]["experiment_name"],
    )
    updated["mlflow"]["tracking_uri"] = os.getenv("MLFLOW_TRACKING_URI", "mlruns")

    return updated


def project_path(relative_path):
    return PROJECT_ROOT / relative_path


def get_data_path():
    config = load_config()
    return project_path(config.get("data", {}).get("disease_data_path", "data/disease_data.csv"))
