"""CSV → Postgres ingest for EpiPulse AI.

Reloads `disease_records` from the synthetic CSV each run. Used by the Airflow
`seed_postgres` task. Full-reload semantics keep the ingest deterministic and
avoid upsert-key handling for the synthetic dataset.
"""

import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import delete

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from database.db import Base, SessionLocal, engine
from database.models import DiseaseRecord
from src.utils.config import get_data_path


BATCH_SIZE = 1000


def reload_disease_records_from_csv(csv_path: Path = None) -> int:
    """Truncate `disease_records` and bulk-load rows from the CSV.

    Returns the number of rows inserted.
    """
    csv_path = Path(csv_path) if csv_path else get_data_path()

    Base.metadata.create_all(engine, tables=[DiseaseRecord.__table__])

    df = pd.read_csv(csv_path, parse_dates=["date"])
    records = [
        {
            "date": row.date.date() if hasattr(row.date, "date") else row.date,
            "region": row.region,
            "cases": int(row.cases) if pd.notna(row.cases) else 0,
            "temperature": float(row.temperature) if pd.notna(row.temperature) else None,
            "humidity": float(row.humidity) if pd.notna(row.humidity) else None,
            "rainfall": float(row.rainfall) if pd.notna(row.rainfall) else None,
        }
        for row in df.itertuples(index=False)
    ]

    session = SessionLocal()
    try:
        session.execute(delete(DiseaseRecord))
        for start in range(0, len(records), BATCH_SIZE):
            session.bulk_insert_mappings(DiseaseRecord, records[start : start + BATCH_SIZE])
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

    return len(records)


if __name__ == "__main__":
    inserted = reload_disease_records_from_csv()
    print(f"Inserted {inserted} disease records")
