# Technical Implementation

## 1. Environment Setup and Ingestion

Use the project-local Python virtual environment for this report. This keeps the assignment dependencies isolated from the system Python installation.

### 1.1 Environment

- Python: 3.14.3
- Virtual environment: `.venv`
- Dependencies: `requirements.txt`

### 1.2 Activate the Environment

```sh
source .venv/bin/activate
```

To leave the environment:

```sh
deactivate
```

### 1.3 Install or Restore Dependencies

Run this from the project directory:

```sh
../../.venv/bin/python -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

The parent project environment at `../../.venv/bin/python` supplies Python 3.14.3 when creating the report environment.

### 1.4 Included Libraries

- Jupyter
- pandas
- dask
- pyspark
- SQLAlchemy
- PyMongo
- kafka-python
- matplotlib
- seaborn

### 1.5 Guideline for Later Steps

All later notebooks and Python scripts for this assignment should use the `.venv` interpreter and the libraries declared in `requirements.txt`. In VS Code, select `.venv/bin/python` as the Python interpreter before creating or running notebooks.

### 1.6 Verify the Setup

```sh
.venv/bin/python -c "import pandas, dask, pyspark, sqlalchemy, pymongo, kafka, matplotlib, seaborn, jupyter; print('All requested packages imported successfully.')"
```

## 2. Main Datasets

The assignment datasets are stored in `datasets/`. The primary dataset for this
assignment is `datasets/alaska_cameras/`, which contains normalized Alaska
camera records in the format expected by BrewER. The raw source files are not
included in this repository.

### 2.1 Alaska Cameras

- `dataset.csv`: normalized camera records with the columns `_id`,
  `description`, `brand`, `model`, `mp`, `optical_zoom`, `digital_zoom`,
  `screen_size`, `price`, and `type`.
- `matches.csv`: ground-truth matching pairs in `l_id,r_id` format. Use this
  file as the gold standard when evaluating entity-resolution results.

The normalized dataset contains 13,583 records and has incomplete attributes.
Missing values are expected, especially for `price` (12,573), `digital_zoom`
(12,779), `optical_zoom` (6,569), `screen_size` (4,967), and `type` (3,118).
`brand` is missing in 714 records and `model` is missing in 741 records. These
missing values reflect the original product records and should be handled or
reported during analysis; they are not evidence that the dataset is invalid.

`datasets/alaska_cameras_small/` is a reduced version for quick development
and testing. It has the same main files with fewer records.

## 3. Pipeline

The code for Steps 1 through 7 is organized in the `pipeline` package. Run the core pipeline from the project directory:

```sh
../../.venv/bin/python run_pipeline.py
```

This command verifies the environment and restarts MongoDB and Kafka,
ingests and cleans the camera CSV data, runs local Spark processing, stores the
camera table in `output/alaska_catalog.sqlite`, publishes Kafka updates, evaluates
product matching, and creates the visualizations. During the main run, the
Spark streaming consumer runs in the background while Kafka publishes records,
then stops before product matching and visualization.

Step 1 clears the contents of `output/` before the run so generated files and
Spark checkpoints are recreated from a clean state. The source files under
`datasets/` are not modified.

### 3.1 Individual Steps

```sh
# Step 1: Environment and Docker service readiness
../../.venv/bin/python -m pipeline.step01_environment

# Step 2: Pandas profiling and Dask cleansing
../../.venv/bin/python -m pipeline.step02_ingest_clean

# Step 3: Local PySpark DataFrame and RDD processing
../../.venv/bin/python -m pipeline.step03_spark_processing

# Step 4: SQLite and MongoDB catalog storage
../../.venv/bin/python -m pipeline.step04_storage

# Step 5.1: Spark Streaming consumer
../../.venv/bin/python -m pipeline.step05_1_spark_streaming

# Step 5.2: Kafka producer, in a second terminal after Step 5.1 starts
../../.venv/bin/python -m pipeline.step05_2_kafka_producer

# Step 6: Token blocking, Levenshtein, Jaccard, Precision/Recall/F1
../../.venv/bin/python -m pipeline.step06_entity_resolution

# Step 7: Price distribution and missingness visualizations
../../.venv/bin/python -m pipeline.step07_visualize
```

PySpark requires a Java runtime. Install a supported JDK and ensure `java -version` succeeds before running either Spark command. The remaining pipeline steps do not require Java.

### 3.2 Kafka Streaming

Kafka configuration is in `docker-compose.yml`. When Docker is available, start the broker and then start the streaming consumer:

```sh
docker compose up -d
../../.venv/bin/python -m pipeline.step05_1_spark_streaming
```

The Structured Streaming consumer requires the Spark Kafka connector on the
classpath. It filters out missing, NaN, zero, and negative prices before
writing. With the consumer running, publish the complete cleaned catalog in a
second terminal:

```sh
../../.venv/bin/python -m pipeline.step05_2_kafka_producer
```

The streaming consumer keeps displaying micro-batches in the console and
upserts only positive-price records into the
`alaska_catalog.streaming_cameras_with_price` MongoDB collection. The collection is
cleared when the streaming consumer starts. Records are upserted by `_id`.

### 3.3 MongoDB

Start MongoDB with Docker Compose:

```sh
docker compose up -d mongodb
../../.venv/bin/python -m pipeline.step04_storage --mongo-uri mongodb://localhost:27017
```

MongoDB data is persisted in the Docker-managed `mongodb_data` volume. Step 4
uses the `cameras` collection when `--mongo-uri` is provided; Step 5 uses the
`streaming_cameras_with_price` collection.

### 3.4 Outputs

- `output/cleaned_alaska_cameras_dataset.csv`: normalized camera records
- `output/cleaned_profile.json`: input profile and missing-value counts
- `output/stats_alaska_cameras.json`: Spark column statistics and token counts
- `output/alaska_catalog.sqlite`: relational camera table
- `output/product_match_candidates.csv`: scored product-pair comparisons
- `output/product_match_metrics.csv`: product-matching precision, recall, and F1 score
- `output/viz_missingness_heatmap.png`: missingness heatmap
- `output/viz_distribution_brand.png`: brand distribution
- `output/viz_distribution_description.png`: description distribution
- `output/viz_distribution_digital_zoom.png`: digital zoom distribution
- `output/viz_distribution_model.png`: model distribution
- `output/viz_distribution_mp.png`: megapixel distribution
- `output/viz_distribution_optical_zoom.png`: optical zoom distribution
- `output/viz_distribution_price.png`: price distribution
- `output/viz_distribution_screen_size.png`: screen-size distribution
- `output/viz_distribution_type.png`: camera-type distribution
