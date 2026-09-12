"""Verify the Python environment required by Assignment 6"""

from __future__ import annotations

import importlib
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path

from pipeline.common import TOPIC_CAMERAS, PROJECT_ROOT, clear_output_dir, get_method_logger, log_duration

REQUIRED_PACKAGES = {
    "pandas": "pandas",
    "dask": "dask",
    "pyspark": "pyspark",
    "sqlalchemy": "sqlalchemy",
    "pymongo": "pymongo",
    "kafka-python": "kafka",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "jupyter": "jupyter",
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("a6.step01_environment")

COMPOSE_SERVICES = ("mongodb", "kafka")
DOCKER_PATHS = (
    "/usr/local/bin/docker",
    "/opt/homebrew/bin/docker",
    "/Applications/Docker.app/Contents/Resources/bin/docker",
)


def ensure_services() -> dict[str, str]:
    """Restart or start the required Docker Compose services."""
    docker = shutil.which("docker") or next((path for path in DOCKER_PATHS if Path(path).exists()), None)
    if docker is None:
        return {service: "docker unavailable" for service in COMPOSE_SERVICES}

    compose = [docker, "compose", "-f", str(PROJECT_ROOT / "docker-compose.yml")]
    start_result = subprocess.run(
        [*compose, "up", "-d", "--force-recreate", *COMPOSE_SERVICES],
        check=False,
    )
    if start_result.returncode != 0:
        return {service: "failed to restart" for service in COMPOSE_SERVICES}

    topic_command = [
        *compose,
        "exec",
        "-T",
        "kafka",
        "/opt/kafka/bin/kafka-topics.sh",
        "--create",
        "--if-not-exists",
        "--topic",
        TOPIC_CAMERAS,
        "--bootstrap-server",
        "localhost:9092",
        "--partitions",
        "1",
        "--replication-factor",
        "1",
    ]
    for _ in range(10):
        topic_result = subprocess.run(topic_command, capture_output=True, text=True, check=False)
        if topic_result.returncode == 0:
            break
        time.sleep(1)
    else:
        return {service: "topic setup failed" for service in COMPOSE_SERVICES}

    return {service: "restarted" for service in COMPOSE_SERVICES}


@log_duration
def run() -> dict[str, str]:
    # Remove outputs from previous runs before the pipeline recreates them.
    clear_output_dir()

    # Check the Python version and required assignment packages.
    versions = {"python": sys.version.split()[0]}
    for package, module_name in REQUIRED_PACKAGES.items():
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError:
            versions[package] = "missing"
        else:
            versions[package] = getattr(module, "__version__", "installed")

    # Log package results and make required Docker services available.
    method_logger = get_method_logger(run)
    for package, version in versions.items():
        method_logger.info("%s: %s", package, version)
    for service, state in ensure_services().items():
        method_logger.info("%s: %s", service, state)
    return versions


if __name__ == "__main__":
    run()
