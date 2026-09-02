import json
import os
import ssl
import sys
import urllib.request
from datetime import date, timedelta

import certifi
import logging
import pandas as pd
from sqlalchemy import create_engine

try:
    from awsglue.utils import getResolvedOptions
except ModuleNotFoundError:
    getResolvedOptions = None

# Database connection settings
DB_HOST = "phi-pipeline-database.c7uq2qsu0b2y.us-east-2.rds.amazonaws.com"
DB_NAME = "dds-8530-5"
DB_USER = "postgres"

logger = logging.getLogger(__name__)

# Directory settings
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data")

# SSL and HTTP settings
SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; Python urllib; +https://www.python.org)",
    "Accept": "*/*",
}


def get_job_args():
    if getResolvedOptions is not None:
        return getResolvedOptions(sys.argv, ["DB_PASSWORD"])
    return {
        "DB_PASSWORD": os.getenv("DB_PASSWORD"),
    }


def extract_db():
    args = get_job_args()
    DB_PASSWORD = args.get("DB_PASSWORD")
    if not DB_PASSWORD:
        logger.error("DB_PASSWORD is not provided in environment variables or AWS job arguments.")
        raise RuntimeError("DB_PASSWORD is not provided in environment variables or AWS job arguments.")

    logger.info("Starting database data extraction.")
    db_url = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:5432/{DB_NAME}"
    engine = create_engine(db_url)

    # Extract data in chunks to handle large datasets
    chunk_size = 10000  # Number of rows per batch
    df_db = pd.DataFrame()
    for chunk in pd.read_sql("SELECT * FROM store_regions", engine, chunksize=chunk_size):
        df_db = pd.concat([df_db, chunk], ignore_index=True)

    logger.info("Finished database data extraction. Total rows: %d", len(df_db))
    return df_db


def extract_api():
    logger.info("Starting API data extraction.")

    start_day = date(2025, 1, 1)
    api_data_frames = []
    max_calls = 25

    for day_offset in range(0, max_calls * 2, 2):
        current_day = start_day + timedelta(days=day_offset)
        next_day = current_day + timedelta(days=2)
        starttime = current_day.isoformat()
        endtime = next_day.isoformat()
        api_url = "https://earthquake.usgs.gov/fdsnws/event/1/query?" f"format=geojson&starttime={starttime}&endtime={endtime}"
        logger.info("API URL: %s", api_url)
        request = urllib.request.Request(api_url, headers=HTTP_HEADERS)
        with urllib.request.urlopen(request, context=SSL_CONTEXT) as response:
            payload = json.loads(response.read().decode("utf-8"))
            logger.info("Rows: %d", len(payload.get("features", [])))

        features = payload.get("features", [])
        if not features:
            continue

        frame = pd.json_normalize(features)
        frame = frame.loc[:, frame.notna().any(axis=0)]
        api_data_frames.append(frame)

        if len(api_data_frames) >= max_calls:
            break

    if not api_data_frames:
        logger.warning("No earthquake data returned from the API.")
        return pd.DataFrame()

    api_data = pd.concat(api_data_frames, ignore_index=True)
    logger.info("Finished API data extraction. Total rows: %d", len(api_data))
    return api_data


def extract_csv():
    logger.info("Starting CSV data extraction.")

    data_s3_url = "https://hoangpdds8530-5.s3.us-east-2.amazonaws.com/retail_data.csv"

    os.makedirs(DATA_DIR, exist_ok=True)

    data_file_path = os.path.join(DATA_DIR, "retail_data.csv")
    if not os.path.exists(data_file_path):
        logger.info("Downloading data from S3 to %s", data_file_path)
        with urllib.request.urlopen(data_s3_url, context=SSL_CONTEXT) as response:
            with open(data_file_path, "wb") as out_file:
                out_file.write(response.read())
    else:
        logger.info("Data file already exists at %s", data_file_path)

    logger.info("Reading CSV file in chunks from %s", data_file_path)
    chunk_size = 10000
    csv_chunks = []
    for chunk in pd.read_csv(data_file_path, chunksize=chunk_size):
        csv_chunks.append(chunk)

    df_csv = pd.concat(csv_chunks, ignore_index=True)
    logger.info("Finished reading CSV file. Total rows: %d", len(df_csv))
    return df_csv


def extract_data():
    csv_data = extract_csv()
    db_data = extract_db()
    api_data = extract_api()
    return csv_data, db_data, api_data
