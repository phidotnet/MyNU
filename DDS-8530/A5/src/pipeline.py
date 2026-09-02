import logging
import time
from contextlib import contextmanager
from logging.handlers import RotatingFileHandler
from pathlib import Path

from .phase_extract import extract_data
from .phase_load import load_transformed_data
from .phase_transform import transform_dataframe
from .ml_train_models import train_models

PROJECT_DIR = Path(__file__).resolve().parents[1]
LOGS_DIR = PROJECT_DIR / "runtime/logs"

logger = logging.getLogger(__name__)


def configure_logging() -> None:
    """Log INFO+ to pipeline.log, ERROR+ to pipeline-error.log, and to stdout."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")

    info_handler = RotatingFileHandler(LOGS_DIR / "pipeline.log", maxBytes=5_000_000, backupCount=3)
    info_handler.setLevel(logging.INFO)
    info_handler.setFormatter(fmt)

    error_handler = RotatingFileHandler(LOGS_DIR / "pipeline-error.log", maxBytes=5_000_000, backupCount=3)
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(fmt)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(fmt)

    logging.basicConfig(level=logging.INFO, handlers=[info_handler, error_handler, console_handler])


@contextmanager
def stage(name: str):
    """Log start/success/failure and elapsed time for one pipeline stage."""
    start = time.perf_counter()
    logger.info("stage=%s status=start", name)
    try:
        yield
    except Exception:
        logger.exception("stage=%s status=failed elapsed_s=%.3f", name, time.perf_counter() - start)
        raise
    else:
        logger.info("stage=%s status=success elapsed_s=%.3f", name, time.perf_counter() - start)


def run_etl_mlops_pipeline() -> None:
    """Run the configured extract, transform, load, and model-training workflow."""
    
    with stage("extract"):
        csv_data, db_data, api_data = extract_data()

    with stage("transform"):
        csv_transformed = transform_dataframe(csv_data)
        db_transformed = transform_dataframe(db_data)
        api_transformed = transform_dataframe(api_data)

    with stage("load"):
        load_transformed_data(csv_transformed, db_transformed, api_transformed)

    with stage("train"):
        train_models(csv_transformed)


if __name__ == "__main__":
    configure_logging()
    try:
        run_etl_mlops_pipeline()
    except Exception:
        logger.exception("pipeline run failed")
        raise
