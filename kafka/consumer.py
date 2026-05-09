import json
import os

from src.alerting.alert_generator import generate_alert
from src.utils.config import load_config


def create_consumer(topic="disease-events", bootstrap_servers=None):
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
        group_id="epipulse-consumer",
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


if __name__ == "__main__":
    consume_events()
