from datetime import datetime
import sys
from pathlib import Path

from airflow.sdk import dag, task

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

@dag(
    dag_id="etl_mlops_pipeline",
    schedule="0 2 * * *",
    start_date=datetime(2026, 8, 1),
    catchup=False,
    tags=["etl", "mlops"],
)
def etl_mlops_pipeline():
    """Schedule the same ETL flow defined in pipeline.py every day."""

    @task
    def run_etl_and_mlops() -> None:
        from src.pipeline import run_etl_mlops_pipeline
        run_etl_mlops_pipeline()

    run_etl_and_mlops()

etl_mlops_pipeline()
