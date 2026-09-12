from __future__ import annotations

import argparse
import itertools
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from time import perf_counter

import pandas as pd
from pyspark.sql import SparkSession, functions as F

if __package__:
    from pipeline.common import *
else:
    from common import *


def tokens(value: str | None) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", normalize_text(value) or ""))


def score_pair(candidates: list[dict[str, object]], spark: SparkSession) -> pd.DataFrame:
    """Score all candidates in one Spark DataFrame operation."""
    if not candidates:
        return pd.DataFrame()

    pairs = spark.createDataFrame(candidates)
    pairs = (
        pairs
        .withColumn("description_left", F.coalesce(F.col("description_left"), F.lit("")))
        .withColumn("description_right", F.coalesce(F.col("description_right"), F.lit("")))
    )
    pairs = pairs.withColumn(
        "description_levenshtein",
        F.levenshtein("description_left", "description_right"),
    )
    pairs = pairs.withColumn(
        "left_tokens",
        F.expr("filter(array_distinct(split(regexp_replace(lower(description_left), '[^a-z0-9]+', ' '), ' ')), token -> token <> '')"),
    ).withColumn(
        "right_tokens",
        F.expr("filter(array_distinct(split(regexp_replace(lower(description_right), '[^a-z0-9]+', ' '), ' ')), token -> token <> '')"),
    )
    pairs = pairs.withColumn("token_union", F.array_union("left_tokens", "right_tokens"))
    pairs = pairs.withColumn(
        "description_jaccard",
        F.when(F.size("token_union") == 0, F.lit(0.0)).otherwise(
            F.size(F.array_intersect("left_tokens", "right_tokens")) / F.size("token_union")
        ),
    )
    pairs = pairs.withColumn(
        "description_similarity",
        1 - F.col("description_levenshtein") / F.greatest(
            F.length("description_left"), F.length("description_right"), F.lit(1)
        ),
    ).withColumn(
        "similarity",
        F.round(
            0.20 * F.col("brand_similarity")
            + 0.20 * F.col("model_similarity")
            + 0.35 * F.col("description_similarity")
            + 0.25 * F.col("description_jaccard"),
            4,
        ),
    )
    return pd.DataFrame(
        row.asDict()
        for row in pairs.select(
            "l_id",
            "r_id",
            "brand_similarity",
            "model_similarity",
            "description_jaccard",
            "description_levenshtein",
            "similarity",
        ).collect()
    )


def expected_pairs() -> set[tuple[str, str]]:
    if not DATASET_MATCHES_FILE.exists():
        raise FileNotFoundError(f"No matches.csv found under {DATASET_DIR}")
    truth = pd.read_csv(DATASET_MATCHES_FILE)
    return {tuple(sorted((str(row.l_id), str(row.r_id)))) for row in truth.itertuples(index=False)}


@log_duration
def run(input_path: Path = DATASET_CLEAN_FILE, threshold: float = 0.55) -> dict[str, float | int]:
    method_logger = get_method_logger(run)

    # Load the cleaned product fields used for matching.
    phase_started = perf_counter()
    method_logger.info("START load_products")
    if not input_path.exists():
        raise FileNotFoundError("Run pipeline.ingest_clean before entity resolution.")
    products = pd.read_csv(input_path).fillna("")[["_id", "description", "brand", "model"]]
    method_logger.info("END load_products records=%s elapsed_seconds=%.4f", len(products), perf_counter() - phase_started)

    # Count tokens so common words can be excluded from blocking.
    phase_started = perf_counter()
    method_logger.info("START count_tokens")
    token_frequency: dict[str, int] = defaultdict(int)
    product_tokens = []
    records = products.to_dict(orient="records")
    for row in records:
        row_tokens = tokens(row["brand"]) | tokens(row["model"]) | tokens(row["description"])
        product_tokens.append(row_tokens)
        for token in row_tokens:
            token_frequency[token] += 1
    method_logger.info("END count_tokens unique_tokens=%s elapsed_seconds=%.4f", len(token_frequency), perf_counter() - phase_started)

    # Group products by informative tokens to reduce pair comparisons.
    phase_started = perf_counter()
    method_logger.info("START build_blocks")
    blocks: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row, row_tokens in zip(records, product_tokens):
        block_tokens = {token for token in row_tokens if token_frequency[token] <= 100}
        for token in block_tokens:
            blocks[token].append(row)
    method_logger.info("END build_blocks blocks=%s elapsed_seconds=%.4f", len(blocks), perf_counter() - phase_started)

    # Score unique cross-source product pairs from each block.
    phase_started = perf_counter()
    method_logger.info("START score_pairs")
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
    spark = (
        SparkSession.builder
        .master("local[*]")
        .appName("AlaskaCameraProductMatching")
        .config("spark.pyspark.python", sys.executable)
        .config("spark.pyspark.driver.python", sys.executable)
        .getOrCreate()
    )
    candidate_keys: set[tuple[str, str]] = set()
    candidates = []
    for records in blocks.values():
        for left, right in itertools.combinations(records, 2):
            left_source = str(left["_id"]).split("//", 1)[0]
            right_source = str(right["_id"]).split("//", 1)[0]
            if left_source == right_source:
                continue
            key = tuple(sorted((left["_id"], right["_id"])))
            if key not in candidate_keys:
                candidate_keys.add(key)
                candidates.append({
                    "l_id": left["_id"],
                    "r_id": right["_id"],
                    "description_left": left["description"],
                    "description_right": right["description"],
                    "brand_similarity": float(left["brand"] == right["brand"]),
                    "model_similarity": float(left["model"] == right["model"]),
                })
    comparisons = score_pair(candidates, spark)
    spark.stop()
    method_logger.info("END score_pairs candidates=%s elapsed_seconds=%.4f", len(comparisons), perf_counter() - phase_started)

    # Compare predicted matches with the Alaska ground-truth pairs.
    phase_started = perf_counter()
    method_logger.info("START evaluate_matches")
    if comparisons.empty:
        comparisons = pd.DataFrame(columns=["l_id", "r_id", "brand_similarity", "model_similarity", "description_jaccard", "description_levenshtein", "similarity"])
    comparisons["predicted_match"] = comparisons["similarity"] >= threshold
    truth = expected_pairs()
    predicted = {tuple(sorted((row.l_id, row.r_id))) for row in comparisons.itertuples() if row.predicted_match}
    true_positives = len(predicted & truth)
    precision = true_positives / len(predicted) if predicted else 0.0
    recall = true_positives / len(truth) if truth else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    method_logger.info("END evaluate_matches predicted=%s elapsed_seconds=%.4f", len(predicted), perf_counter() - phase_started)

    # Save detailed pair scores and summary evaluation metrics.
    phase_started = perf_counter()
    method_logger.info("START save_outputs")
    ensure_output_dir()
    comparisons.to_csv(PRODUCT_MATCH_CANDIDATES_FILE, index=False)
    metrics = {
        "candidate_pairs": len(comparisons),
        "expected_pairs": len(truth),
        "predicted_pairs": len(predicted),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
    }
    pd.DataFrame([metrics]).to_csv(PRODUCT_MATCH_METRICS_FILE, index=False)
    method_logger.info("END save_outputs elapsed_seconds=%.4f", perf_counter() - phase_started)
    method_logger.info("metrics=%s", metrics)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resolve product entities using token blocking and string similarity.")
    parser.add_argument("--input", type=Path, default=DATASET_CLEAN_FILE)
    parser.add_argument("--threshold", type=float, default=0.55)
    arguments = parser.parse_args()
    run(arguments.input, arguments.threshold)
