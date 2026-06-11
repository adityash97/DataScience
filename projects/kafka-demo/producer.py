from kafka import KafkaProducer
import json
import time
from datetime import datetime

producer = KafkaProducer(
    bootstrap_servers="localhost:9092",
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)

counter = 1

while True:
    log = {
        "id": counter,
        "timestamp": datetime.now().isoformat(),
        "level": "INFO",
        "message": f"Log message {counter}"
    }

    producer.send("logs", log)
    producer.flush()

    print("Produced:", log)

    counter += 1
    time.sleep(2)