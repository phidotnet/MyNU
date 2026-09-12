from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from time import perf_counter

from pyspark.sql import SparkSession, functions as F

if __package__:
    from pipeline.common import *
else:
    from common import *

def clean_text(text: str | None) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", (text or "").lower()).strip()


def stats_numeric(products, column: str) -> dict[str, object]:
    values = products.select(column).where(F.col(column).isNotNull())
    count = values.count()
    total = products.count()
    result = values.select(
        F.min(column).alias("min"),
        F.max(column).alias("max"),
        F.avg(column).alias("mean"),
        F.expr(f"percentile_approx(`{column}`, 0.5)").alias("median"),
    ).first().asDict()
    return {
        "type": "numeric",
        "count": count,
        "missing": total - count,
        "distinct": values.distinct().count(),
        **{key: float(value) if value is not None else None for key, value in result.items()},
    }


def stats_categorical(products, column: str) -> dict[str, object]:
    values = products.select(column).where(F.col(column).isNotNull())
    top_values = [row.asDict() for row in values.groupBy(column).count().orderBy(F.desc("count")).limit(10).collect()]
    return {
        "type": "categorical",
        "count": values.count(),
        "distinct": values.distinct().count(),
        "top_values": top_values,
    }


def stats_token_counts(products, column: str = "description") -> dict[str, object]:
    tokens = (
        products.select(column).rdd.flatMap(lambda row: clean_text(row[column]).split())
        .filter(bool)
        .map(lambda token: (token, 1))
        .reduceByKey(lambda left, right: left + right)
        .sortBy(lambda item: (-item[1], item[0]))
        .take(25)
    )
    return {"type": "token_counts", "top_tokens": [{"token": token, "count": count} for token, count in tokens]}


def numeric_column(products, column: str) -> bool:
    values = products.select(F.col(column)).where(F.col(column).isNotNull())
    non_null_count = values.count()
    if not non_null_count:
        return False
    numeric_count = values.select(F.expr(f"try_cast(`{column}` AS DOUBLE)").alias("value")).where(F.col("value").isNotNull()).count()
    return numeric_count == non_null_count


@log_duration
def run(input_path: Path = DATASET_CLEAN_FILE) -> dict[str, object]:
    if not input_path.exists():
        raise FileNotFoundError("Run pipeline.ingest_clean before Spark processing.")
    ensure_output_dir()
    
    spark = SparkSession.builder.master("local[*]").appName("AlaskaCameraSparkProcessing").getOrCreate()
    started = perf_counter()
    products = spark.read.option("header", True).option("inferSchema", True).csv(str(input_path))
    products = products.withColumn("price", F.expr("try_cast(price AS DOUBLE)"))
    rdd_started = perf_counter()
    numeric_types = {"byte", "short", "int", "long", "float", "double", "decimal"}
    column_stats = {}
    for column, data_type in products.dtypes:
        if column == "_id":
            continue
        if column == "description":
            column_stats[column] = stats_token_counts(products)
        elif any(data_type.startswith(numeric_type) for numeric_type in numeric_types) or numeric_column(products, column):
            if not any(data_type.startswith(numeric_type) for numeric_type in numeric_types):
                products = products.withColumn(column, F.expr(f"try_cast(`{column}` AS DOUBLE)"))
            column_stats[column] = stats_numeric(products, column)
        else:
            column_stats[column] = stats_categorical(products, column)

    metrics = {
        "records": products.count(),
        "dataframe_seconds": round(perf_counter() - started, 4),
        "rdd_seconds": round(perf_counter() - rdd_started, 4),
        "columns": column_stats,
    }
    
    DATASET_STATS_FILE.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    get_method_logger(run).info("%s", metrics)
    spark.stop()
    
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PySpark DataFrame and RDD transformations.")
    parser.add_argument("--input", type=Path, default=DATASET_CLEAN_FILE)
    arguments = parser.parse_args()
    run(arguments.input)