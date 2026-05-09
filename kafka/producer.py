import json
import os

import pandas as pd

from src.utils.config import get_data_path, load_config


def create_producer(bootstrap_servers=None):
    try:
        from kafka import KafkaProducer
    except ImportError as exc:
        raise ImportError(
            "KafkaProducer is unavailable. Install kafka-python and run this module outside "
            "the local package import path if the project kafka/ folder shadows the dependency."
        ) from exc

    config = load_config().get("kafka", {})
    return KafkaProducer(
        bootstrap_servers=bootstrap_servers
        or os.getenv("KAFKA_BOOTSTRAP_SERVERS", config.get("bootstrap_servers", "localhost:9094")),
        value_serializer=lambda value: json.dumps(value).encode("utf-8"),
    )


def send_disease_event(event, topic=None, producer=None):
    topic = topic or load_config().get("kafka", {}).get("topic", "disease-events")
    kafka_producer = producer or create_producer()
    kafka_producer.send(topic, event)
    kafka_producer.flush()
    return event


def stream_disease_data(limit=25, topic=None, producer=None):
    df = pd.read_csv(get_data_path()).head(limit)
    events = df.to_dict(orient="records")

    kafka_producer = producer or create_producer()
    for event in events:
        send_disease_event(event, topic=topic, producer=kafka_producer)

    return events


if __name__ == "__main__":
    sent_events = stream_disease_data(limit=10)
    print(f"Sent {len(sent_events)} disease events")
