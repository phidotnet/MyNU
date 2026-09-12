from __future__ import annotations

import logging
import os
import re
import shutil
import sys
from functools import wraps
from pathlib import Path
from time import perf_counter
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("a6")

BREW_JAVA_HOME = Path("/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home")
os.environ["PATH"] = ":".join(dict.fromkeys((os.environ.get("PATH", "").split(":") + ["/usr/local/bin", "/usr/bin", "/bin"])))
if not os.environ.get("JAVA_HOME") and (BREW_JAVA_HOME / "bin/java").exists():
    os.environ["JAVA_HOME"] = str(BREW_JAVA_HOME)
    os.environ["PATH"] = f"{BREW_JAVA_HOME / 'bin'}:{os.environ.get('PATH', '')}"
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "datasets/alaska_cameras"
OUTPUT_DIR = PROJECT_ROOT / "output"
DATASET_RAW_FILE = DATASET_DIR / "dataset.csv"
DATASET_MATCHES_FILE = DATASET_DIR / "matches.csv"
DATASET_CLEAN_FILE = OUTPUT_DIR / "cleaned_alaska_cameras_dataset.csv"
DATASET_STATS_FILE = OUTPUT_DIR / "stats_alaska_cameras.json"
PRODUCT_MATCH_CANDIDATES_FILE = OUTPUT_DIR / "product_match_candidates.csv"
PRODUCT_MATCH_METRICS_FILE = OUTPUT_DIR / "product_match_metrics.csv"
DB_SQLITE = OUTPUT_DIR / "alaska_catalog.sqlite"
TOPIC_CAMERAS = "alaska_cameras_updates"
MONGO_URI = "mongodb://localhost:27017"
MONGO_DATABASE = "alaska_catalog"
MONGO_STREAMING_COLLECTION = "streaming_cameras_with_price"
SPARK_CHECKPOINT = str(OUTPUT_DIR / "spark_checkpoints" / TOPIC_CAMERAS)
STREAM_DRAIN_SECONDS = 5
STREAM_BATCH_TIMEOUT_SECONDS = 120
BOOTSTRAP_SERVER = "localhost:9092"
SPARK_KAFKA_PACKAGE = "org.apache.spark:spark-sql-kafka-0-10_2.13:4.2.0"

def get_method_logger(function):
    return logging.getLogger(f"{function.__module__}.{function.__qualname__}")


def log_duration(function):
    @wraps(function)
    def timed(*args, **kwargs):
        method_logger = get_method_logger(function)
        method_logger.info("START")
        started = perf_counter()
        try:
            return function(*args, **kwargs)
        finally:
            method_logger.info("END elapsed_seconds=%.4f", perf_counter() - started)

    return timed

def ensure_output_dir() -> Path:
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    return OUTPUT_DIR


def clear_output_dir() -> Path:
    if OUTPUT_DIR.exists():
        for item in OUTPUT_DIR.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
    else:
        OUTPUT_DIR.mkdir(parents=True)
    return OUTPUT_DIR

# Normalize text by stripping whitespace, converting to lowercase, and handling special cases
def normalize_text(value: Any) -> str | None:
    if value is None:
        return None
    value = re.sub(r"\s+", " ", str(value)).strip()
    return None if value.lower() in {"nan", "<na>", "nat"} else value.lower() or None

# Parse a price string and return its float value, or None if not parseable
def parse_price(value: Any) -> float | None:
    if value is None:
        return None
    match = re.search(r"\d+(?:\.\d+)?", str(value).replace(",", ""))
    return float(match.group()) if match else None
