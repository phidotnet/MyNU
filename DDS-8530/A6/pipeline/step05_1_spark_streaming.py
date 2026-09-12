from __future__ import annotations

import argparse

from pymongo import MongoClient, ReplaceOne
from pyspark.sql import SparkSession, functions as F, types as T

if __package__:
    from pipeline.common import *
else:
    from common import *

@log_duration
def run(bootstrap_servers: str = BOOTSTRAP_SERVER, topic: str = TOPIC_CAMERAS, checkpoint: str = SPARK_CHECKPOINT, mongo_uri: str = MONGO_URI, stop_event=None, ready_event=None, batch_event=None) -> None:
    method_logger = get_method_logger(run)
    spark = (
        SparkSession.builder
        .appName("AlaskaCameraKafkaStreaming")
        .config("spark.jars.packages", SPARK_KAFKA_PACKAGE)
        .getOrCreate()
    )
    schema = T.StructType([
        T.StructField("_id", T.StringType()),
        T.StructField("description", T.StringType()),
        T.StructField("brand", T.StringType()),
        T.StructField("model", T.StringType()),
        T.StructField("mp", T.DoubleType()),
        T.StructField("optical_zoom", T.DoubleType()),
        T.StructField("digital_zoom", T.DoubleType()),
        T.StructField("screen_size", T.DoubleType()),
        T.StructField("price", T.DoubleType()),
        T.StructField("type", T.StringType()),
    ])
    stream = spark.readStream.format("kafka").option("kafka.bootstrap.servers", bootstrap_servers).option("subscribe", topic).load()
    json_value = F.regexp_replace(
        F.col("value").cast("string"),
        r'(?<![A-Za-z0-9_"\\])NaN(?![A-Za-z0-9_"\\])',
        "null",
    )
    products = (
        stream.select(F.from_json(json_value, schema).alias("camera"))
        .select("camera.*")
        .filter(F.col("price").isNotNull() & ~F.isnan("price") & (F.col("price") > 0))
    )
    mongo_client = MongoClient(mongo_uri)
    database = mongo_client[MONGO_DATABASE]
    if MONGO_STREAMING_COLLECTION not in database.list_collection_names():
        database.create_collection(MONGO_STREAMING_COLLECTION)
    collection = database[MONGO_STREAMING_COLLECTION]
    deleted_count = collection.delete_many({}).deleted_count
    method_logger.info("Cleared %s existing streaming records", deleted_count)

    def write_batch(batch, batch_id: int) -> None:
        batch.cache()
        try:
            batch.show(truncate=False)
            records = [row.asDict(recursive=True) for row in batch.collect()]
        finally:
            batch.unpersist()
        if records:
            collection.bulk_write(
                [ReplaceOne({"_id": record["_id"]}, record, upsert=True) for record in records],
                ordered=False,
            )
            if batch_event is not None:
                batch_event.set()
        method_logger.info("Streaming batch %s stored records=%s", batch_id, len(records))

    query = products.writeStream.foreachBatch(write_batch).option("checkpointLocation", checkpoint).outputMode("append").start()
    query.processAllAvailable()
    if ready_event is not None:
        ready_event.set()
    try:
        if stop_event is None:
            query.awaitTermination()
        else:
            while not stop_event.is_set():
                query.awaitTermination(1)
            query.stop()
    finally:
        mongo_client.close()
        spark.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Consume catalog updates with Spark Structured Streaming.")
    parser.add_argument("--bootstrap-servers", default=BOOTSTRAP_SERVER)
    parser.add_argument("--topic", default=TOPIC_CAMERAS)
    parser.add_argument("--checkpoint", default=SPARK_CHECKPOINT)
    parser.add_argument("--mongo-uri", default=MONGO_URI)
    arguments = parser.parse_args()
    run(arguments.bootstrap_servers, arguments.topic, arguments.checkpoint, arguments.mongo_uri)