from __future__ import annotations

import argparse
import json
from pathlib import Path

import dask.dataframe as dd
import pandas as pd

if __package__:
    from pipeline.common import *
else:
    from common import *


def dataset_path(dataset_dir: Path = DATASET_DIR) -> Path:
    path = DATASET_RAW_FILE if dataset_dir == DATASET_DIR else dataset_dir / "dataset.csv"
    if not path.exists():
        raise FileNotFoundError(f"No dataset.csv found under {dataset_dir}")
    return path


def read_raw_frame(dataset_dir: Path = DATASET_DIR) -> pd.DataFrame:
    frame = pd.read_csv(dataset_path(dataset_dir))
    expected_columns = ["_id", "description", "brand", "model", "mp", "optical_zoom", "digital_zoom", "screen_size", "price", "type"]
    missing_columns = sorted(set(expected_columns) - set(frame.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")
    return frame[expected_columns]


def clean_frame(frame: pd.DataFrame) -> pd.DataFrame:
    cleaned = frame.copy()
    for column in ("description", "brand", "model", "type"):
        cleaned[column] = cleaned[column].map(normalize_text)
    for column in ("mp", "optical_zoom", "digital_zoom", "screen_size", "price"):
        cleaned[column] = cleaned[column].map(parse_price)
    cleaned = cleaned.drop_duplicates(subset=["_id"])
    return cleaned.sort_values("_id").reset_index(drop=True)


@log_duration
def run(dataset_dir: Path = DATASET_DIR) -> tuple[pd.DataFrame, dict[str, object]]:
    raw_frame = read_raw_frame(dataset_dir)
    dask_frame = dd.from_pandas(raw_frame, npartitions=min(8, max(1, len(raw_frame) // 2500)))
    cleaned = dask_frame.map_partitions(clean_frame, meta=clean_frame(raw_frame.head(0))).compute()
    cleaned = clean_frame(cleaned)

    output_dir = ensure_output_dir()
    cleaned.to_csv(DATASET_CLEAN_FILE, index=False)
    profile = {
        "input_records": len(raw_frame),
        "output_records": len(cleaned),
        "partitions": dask_frame.npartitions,
        "missing_values_before": raw_frame.isna().sum().to_dict(),
        "missing_values_after": cleaned.isna().sum().to_dict(),
        "missing_percent_before": (raw_frame.isna().mean() * 100).round(2).to_dict(),
        "missing_percent_after": (cleaned.isna().mean() * 100).round(2).to_dict(),
    }
    (output_dir / "cleaned_profile.json").write_text(json.dumps(profile, indent=2), encoding="utf-8")
    get_method_logger(run).info("Wrote %s cleaned records using %s Dask partitions.", len(cleaned), profile["partitions"])
    
    return cleaned, profile


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest and clean the Alaska camera CSV dataset.")
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    arguments = parser.parse_args()
    frame, summary = run(arguments.dataset_dir)