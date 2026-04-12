"""
Avro serialization helpers for Kafka messages.

Provides configured producer/consumer factories with schema registry
integration for the liquidwar5-ai evolution pipeline.
"""

import json
from pathlib import Path

from confluent_kafka import Producer, Consumer
from confluent_kafka.schema_registry import SchemaRegistryClient
from confluent_kafka.schema_registry.avro import (
    AvroSerializer,
    AvroDeserializer,
)
from confluent_kafka.serialization import (
    SerializationContext,
    MessageField,
    StringSerializer,
    StringDeserializer,
)

SCHEMA_DIR = Path(__file__).parent / "schemas"

TOPIC_JOBS = "ml.liquidwar5.game-jobs"
TOPIC_RESULTS = "ml.liquidwar5.game-results"
TOPIC_STATE = "ml.liquidwar5.evolution-state"


def _load_schema(name: str) -> str:
    """Load an Avro schema file, resolving nested references."""
    # Load the shared AIParams schema
    params_schema = json.loads((SCHEMA_DIR / "ai_params.avsc").read_text())

    schema = json.loads((SCHEMA_DIR / f"{name}.avsc").read_text())

    # Replace string references to AIParams with the inline schema.
    # Also handle nested records that reference AIParams.
    return json.dumps(_resolve_refs(schema, {"com.liquidwar5.ai.AIParams": params_schema}))


def _resolve_refs(obj, named_types: dict):
    """Recursively resolve named type references in an Avro schema."""
    if isinstance(obj, str):
        return named_types.get(obj, obj)
    if isinstance(obj, list):
        return [_resolve_refs(item, named_types) for item in obj]
    if isinstance(obj, dict):
        # If this is a record, register it as a named type
        if obj.get("type") == "record" and "name" in obj:
            full_name = obj.get("namespace", "") + "." + obj["name"] if obj.get("namespace") else obj["name"]
            # Resolve fields first
            resolved = {k: _resolve_refs(v, named_types) for k, v in obj.items()}
            named_types[full_name] = resolved
            return resolved
        return {k: _resolve_refs(v, named_types) for k, v in obj.items()}
    return obj


def create_schema_registry(url: str) -> SchemaRegistryClient:
    return SchemaRegistryClient({"url": url})


def create_avro_producer(bootstrap_servers: str,
                         schema_registry_url: str,
                         topic: str) -> tuple:
    """Create a Producer and AvroSerializer for a topic.

    Returns (producer, serializer, key_serializer).
    """
    sr_client = create_schema_registry(schema_registry_url)

    schema_map = {
        TOPIC_JOBS: "game_job",
        TOPIC_RESULTS: "game_result",
        TOPIC_STATE: "evolution_state",
    }
    schema_str = _load_schema(schema_map[topic])

    serializer = AvroSerializer(
        sr_client,
        schema_str,
        conf={"auto.register.schemas": True},
    )

    producer = Producer({
        "bootstrap.servers": bootstrap_servers,
        "compression.type": "snappy",
        "linger.ms": 5,
        "batch.num.messages": 100,
    })

    key_serializer = StringSerializer("utf_8")

    return producer, serializer, key_serializer


def create_avro_consumer(bootstrap_servers: str,
                         schema_registry_url: str,
                         topic: str,
                         group_id: str) -> tuple:
    """Create a Consumer and AvroDeserializer for a topic.

    Returns (consumer, deserializer, key_deserializer).
    """
    sr_client = create_schema_registry(schema_registry_url)

    schema_map = {
        TOPIC_JOBS: "game_job",
        TOPIC_RESULTS: "game_result",
        TOPIC_STATE: "evolution_state",
    }
    schema_str = _load_schema(schema_map[topic])

    deserializer = AvroDeserializer(
        sr_client,
        schema_str,
    )

    consumer = Consumer({
        "bootstrap.servers": bootstrap_servers,
        "group.id": group_id,
        "auto.offset.reset": "earliest",
        "enable.auto.commit": True,
        "max.poll.interval.ms": 600000,
    })
    consumer.subscribe([topic])

    key_deserializer = StringDeserializer("utf_8")

    return consumer, deserializer, key_deserializer


def produce_avro(producer, serializer, key_serializer, topic: str,
                 key: str, value: dict):
    """Produce an Avro-serialized message."""
    producer.produce(
        topic=topic,
        key=key_serializer(key),
        value=serializer(
            value,
            SerializationContext(topic, MessageField.VALUE),
        ),
    )


def consume_avro(consumer, deserializer, key_deserializer,
                 timeout: float = 1.0) -> tuple:
    """Consume and deserialize one Avro message.

    Returns (key, value) or (None, None) if no message or deserialization fails.
    """
    msg = consumer.poll(timeout)
    if msg is None:
        return None, None
    if msg.error():
        return None, None

    try:
        key = key_deserializer(msg.key()) if msg.key() else None
        value = deserializer(
            msg.value(),
            SerializationContext(msg.topic(), MessageField.VALUE),
        )
        return key, value
    except Exception:
        # Skip messages that weren't produced with Avro serializer
        return None, None
