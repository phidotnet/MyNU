from __future__ import annotations

import multiprocessing
import time

from pipeline import (
    step01_environment,
    step02_ingest_clean,
    step03_spark_processing,
    step04_storage,
    step05_1_spark_streaming,
    step05_2_kafka_producer,
    step06_entity_resolution,
    step07_visualize,
)
from pipeline.common import STREAM_BATCH_TIMEOUT_SECONDS, STREAM_DRAIN_SECONDS


def run_streaming(stop_event, ready_event, batch_event) -> None:
    step05_1_spark_streaming.run(stop_event=stop_event, ready_event=ready_event, batch_event=batch_event)


def wait_for_streaming_batch(streaming_process, batch_event) -> None:
    deadline = time.monotonic() + STREAM_BATCH_TIMEOUT_SECONDS
    while not batch_event.is_set() and streaming_process.is_alive() and time.monotonic() < deadline:
        time.sleep(1)
    if batch_event.is_set():
        return
    if not streaming_process.is_alive():
        raise RuntimeError("Step 5.1 Spark Streaming stopped before storing a Kafka batch.")
    raise RuntimeError("Step 5.1 did not store a non-empty Kafka batch in MongoDB.")


def stop_streaming(streaming_process, stop_event) -> None:
    stop_event.set()
    streaming_process.join(timeout=30)
    if streaming_process.is_alive():
        streaming_process.terminate()
        streaming_process.join()


def main() -> None:
    step01_environment.run()
    step02_ingest_clean.run()
    step03_spark_processing.run()
    step04_storage.run()
    stop_event = multiprocessing.Event()
    ready_event = multiprocessing.Event()
    batch_event = multiprocessing.Event()
    streaming_process = multiprocessing.Process(target=run_streaming, args=(stop_event, ready_event, batch_event))
    streaming_process.start()
    try:
        if not ready_event.wait(timeout=60):
            raise RuntimeError("Step 5.1 Spark Streaming did not start within 60 seconds.")
        if not streaming_process.is_alive():
            raise RuntimeError("Step 5.1 Spark Streaming stopped before Kafka publishing started.")
        step05_2_kafka_producer.run()
        wait_for_streaming_batch(streaming_process, batch_event)
        time.sleep(STREAM_DRAIN_SECONDS)
    finally:
        stop_streaming(streaming_process, stop_event)
    step06_entity_resolution.run()
    step07_visualize.run()


if __name__ == "__main__":
    main()