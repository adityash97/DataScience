from kafka import KafkaConsumer
import json

consumer = KafkaConsumer(
    "logs",
    bootstrap_servers="localhost:9092",
    auto_offset_reset="earliest",
    group_id="log-consumers",
    value_deserializer=lambda m: json.loads(m.decode("utf-8"))
)

print("Waiting for messages...\n")

for message in consumer:
    print("Consumed:", message.value)