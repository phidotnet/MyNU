import logging
import os
from collections.abc import Mapping

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine, URL

DB_HOST = "phi-pipeline-database.c7uq2qsu0b2y.us-east-2.rds.amazonaws.com"
DB_NAME = "dds-8530-5"
DB_USER = "postgres"

TABLE_NAMES = {
    "retail_transactions": "transformed_retail_transactions",
    "store_regions": "transformed_store_regions",
    "earthquake": "transformed_earthquake",
}

logger = logging.getLogger(__name__)


def create_postgres_engine() -> Engine:
    """Create a PostgreSQL SQLAlchemy engine using DB_PASSWORD from the environment."""
    db_password = os.getenv("DB_PASSWORD")
    if not db_password:
        raise RuntimeError("DB_PASSWORD is not provided in the environment.")

    database_url = URL.create(
        "postgresql+psycopg2",
        username=DB_USER,
        password=db_password,
        host=DB_HOST,
        port=5432,
        database=DB_NAME,
    )
    return create_engine(database_url)


def load_dataframe(
    dataframe: pd.DataFrame,
    table_name: str,
    engine: Engine,
    chunksize: int = 10_000,
) -> None:
    """Replace a database table with the provided transformed DataFrame."""
    logger.info("Loading %d rows into %s.", len(dataframe), table_name)
    dataframe.to_sql(
        table_name,
        engine,
        if_exists="replace",
        index=False,
        chunksize=chunksize,
        method="multi",
    )
    logger.info("Finished loading %s.", table_name)


def load_transformed_data(
    retail_transactions: pd.DataFrame,
    store_regions: pd.DataFrame,
    earthquakes: pd.DataFrame,
    engine: Engine | None = None,
) -> None:
    """Load all transformed datasets into their PostgreSQL destination tables."""
    database_engine = engine if engine is not None else create_postgres_engine()
    datasets: Mapping[str, pd.DataFrame] = {
        TABLE_NAMES["retail_transactions"]: retail_transactions,
        TABLE_NAMES["store_regions"]: store_regions,
        TABLE_NAMES["earthquake"]: earthquakes,
    }

    with database_engine.begin() as connection:
        for table_name, dataframe in datasets.items():
            load_dataframe(dataframe, table_name, connection)
