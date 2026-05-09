import os
import sys
from pathlib import Path

from sqlalchemy import create_engine

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config


def get_database_url():
    config = load_config()
    return os.getenv("DATABASE_URL", config.get("database", {}).get("url"))


def get_engine():
    database_url = get_database_url()
    if not database_url:
        raise ValueError("Database URL is not configured")
    return create_engine(database_url)


if __name__ == "__main__":
    engine = get_engine()
