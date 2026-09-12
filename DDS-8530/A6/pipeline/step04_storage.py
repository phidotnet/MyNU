from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from pymongo import MongoClient
from sqlalchemy import create_engine

if __package__:
    from pipeline.common import *
else:
    from common import *

def write_sqlite(products: pd.DataFrame, database_path: Path) -> int:
    engine = create_engine(f"sqlite:///{database_path}")
    products.to_sql("cameras", engine, if_exists="replace", index=False)
    return len(products)


def write_mongodb(products: pd.DataFrame, mongo_uri: str, database: str = MONGO_DATABASE) -> int:
    documents = products.to_dict(orient="records")
    collection = MongoClient(mongo_uri)[database]["cameras"]
    collection.delete_many({})
    return len(collection.insert_many(documents).inserted_ids) if documents else 0


@log_duration
def run(input_path: Path = DATASET_CLEAN_FILE, mongo_uri: str | None = MONGO_URI) -> dict[str, int]:
    if not input_path.exists():
        raise FileNotFoundError("Run pipeline.ingest_clean before storage integration.")
    output_dir = ensure_output_dir()
    
    products = pd.read_csv(input_path)
    results = {"sqlite_records": write_sqlite(products, DB_SQLITE)}
    if mongo_uri:
        results["mongodb_records"] = write_mongodb(products, mongo_uri)
    
    get_method_logger(run).info("%s", results)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Store cleaned products in SQLite and optionally MongoDB.")
    parser.add_argument("--input", type=Path, default=DATASET_CLEAN_FILE)
    parser.add_argument("--mongo-uri", default=MONGO_URI, help="MongoDB URI, for example mongodb://localhost:27017")
    arguments = parser.parse_args()
    run(arguments.input, arguments.mongo_uri)
    