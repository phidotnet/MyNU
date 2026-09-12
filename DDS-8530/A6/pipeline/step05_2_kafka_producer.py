from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import sleep

import pandas as pd
from kafka import KafkaProducer

if __package__:
    from pipeline.common import *
else:
    from common import *

DELAY_DEFAULT = 0.00025

@log_duration
def run(bootstrap_servers: str = BOOTSTRAP_SERVER, topic: str = TOPIC_CAMERAS, input_path: Path = DATASET_CLEAN_FILE, delay: float = DELAY_DEFAULT) -> int:
    products = pd.read_csv(input_path)
    producer = KafkaProducer(bootstrap_servers=bootstrap_servers, value_serializer=lambda value: json.dumps(value).encode("utf-8"))
    for record in products.to_dict(orient="records"):
        producer.send(topic, record)
        #sleep(delay) # Testing purposes: slow down the publishing to avoid overwhelming the consumer
    producer.flush()
    get_method_logger(run).info("Published %s records.", len(products))
    return len(products)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Publish cleaned catalog records to Kafka.")
    parser.add_argument("--bootstrap-servers", default=BOOTSTRAP_SERVER)
    parser.add_argument("--topic", default=TOPIC_CAMERAS)
    parser.add_argument("--input", type=Path, default=DATASET_CLEAN_FILE)
    parser.add_argument("--delay", type=float, default=DELAY_DEFAULT)
    arguments = parser.parse_args()
    run(arguments.bootstrap_servers, arguments.topic, arguments.input, arguments.delay)