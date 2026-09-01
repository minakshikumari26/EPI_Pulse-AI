import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.alerting.alert_generator import generate_alert
from src.utils.config import load_config


def create_consumer(topic="disease-events", bootstrap_servers=None, group_id="epipulse-consumer"):
    try:
        from kafka import KafkaConsumer
    except ImportError as exc:
        raise ImportError(
            "KafkaConsumer is unavailable. Install kafka-python and run this module outside "
            "the local package import path if the project kafka/ folder shadows the dependency."
        ) from exc

    config = load_config().get("kafka", {})
    return KafkaConsumer(
        topic or config.get("topic", "disease-events"),
        bootstrap_servers=bootstrap_servers
        or os.getenv("KAFKA_BOOTSTRAP_SERVERS", config.get("bootstrap_servers", "localhost:9094")),
        value_deserializer=lambda value: json.loads(value.decode("utf-8")),
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        group_id=group_id,
        consumer_timeout_ms=10000,
    )


def consume_events(topic="disease-events"):
    consumer = create_consumer(topic=topic)
    for message in consumer:
        event = process_event(message.value)
        print(event)


def process_event(event):
    event = dict(event)
    event["alert"] = generate_alert(
        cases=event.get("cases"),
        risk_score=event.get("risk_score"),
        z_score=event.get("z_score"),
        growth_rate=event.get("growth_rate"),
    )
    return event


def consume_and_persist(max_messages: int = 100, topic: str = "disease-events") -> int:
    """Consume up to `max_messages` events and persist them to Postgres.

    Used by the Airflow `consume_to_postgres` task — bounded so the task exits
    cleanly instead of running forever. Returns the number of rows inserted.
    """
    import pandas as pd

    from database.db import Base, SessionLocal, engine
    from database.models import DiseaseRecord

    Base.metadata.create_all(engine, tables=[DiseaseRecord.__table__])

    consumer = create_consumer(topic=topic, group_id="epipulse-airflow-consumer")

    records = []
    try:
        for message in consumer:
            event = message.value
            try:
                event_date = pd.to_datetime(event.get("date")).date()
            except Exception:
                event_date = None
            records.append(
                {
                    "date": event_date,
                    "region": event.get("region"),
                    "cases": int(event["cases"]) if event.get("cases") is not None else 0,
                    "temperature": event.get("temperature"),
                    "humidity": event.get("humidity"),
                    "rainfall": event.get("rainfall"),
                }
            )
            if len(records) >= max_messages:
                break
    finally:
        consumer.close()

    if not records:
        return 0

    session = SessionLocal()
    try:
        session.bulk_insert_mappings(DiseaseRecord, records)
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

    return len(records)


if __name__ == "__main__":
    consume_events()
