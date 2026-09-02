import logging
import re
import warnings
from typing import List, Sequence

import pandas as pd

try:
    import dask.dataframe as dd
except ImportError:  # pragma: no cover
    dd = None

logger = logging.getLogger(__name__)

NORMALIZATION_EXCLUSIONS = {"average_purchase_value"}


def format_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Format column names to a consistent format."""
    cleaned = df.copy()
    cleaned.columns = [re.sub(r"[^0-9a-zA-Z]+", "_", str(col)).strip("_").lower() for col in cleaned.columns]
    logger.info("Formatted column names: %s", cleaned.columns.tolist())
    return cleaned


def _as_scalar_fill_value(value: object) -> object:
    """Convert list-like fill candidates into a scalar value accepted by pandas.fillna."""
    if value is None:
        return "unknown"
    if isinstance(value, (str, bytes, int, float, bool)):
        return value
    if isinstance(value, (list, tuple, set, dict)):
        return str(value)
    if hasattr(value, "tolist"):
        try:
            return str(value.tolist())
        except Exception:
            pass
    if hasattr(value, "__iter__"):
        try:
            return str(list(value))
        except Exception:
            pass
    return value


def handle_missing_values_pandas(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values using realistic defaults without introducing synthetic medians."""
    logger.info("Handling missing values for DataFrame with %d rows and %d columns.", len(df), len(df.columns))
    cleaned = df.copy()

    for column in cleaned.columns:
        # Count missing values in the column
        missing_count = cleaned[column].isna().sum()

        if missing_count == 0:
            continue
        else:
            logger.info("Missing values in column '%s' (%s): %d", column, cleaned[column].dtype, missing_count)

        if pd.api.types.is_numeric_dtype(cleaned[column]):
            replacement = 0
        else:
            replacement = "unknown"

        cleaned[column] = cleaned[column].fillna(replacement)

    return cleaned


def normalize_numeric_columns_pandas(df: pd.DataFrame, columns: Sequence[str] | None = None) -> pd.DataFrame:
    """Standardize numeric columns by z-score normalization."""
    logger.info("Normalizing numeric columns for DataFrame with %d rows and %d columns.", len(df), len(df.columns))
    normalized = df.copy()
    target_columns = (
        columns
        if columns is not None
        else [column for column in normalized.columns if pd.api.types.is_numeric_dtype(normalized[column]) and column not in NORMALIZATION_EXCLUSIONS]
    )

    for column in target_columns:
        logger.info("Normalizing column '%s' (%s).", column, normalized[column].dtype)
        if normalized[column].std(ddof=0) == 0:
            normalized[column] = 0
        else:
            normalized[column] = (normalized[column] - normalized[column].mean()) / normalized[column].std(ddof=0)

    return normalized


def extract_features_pandas(df: pd.DataFrame) -> pd.DataFrame:
    """Create transaction and earthquake features when source columns exist."""
    logger.info("Extracting features for DataFrame with %d rows and %d columns.", len(df), len(df.columns))
    extracted = df.copy()

    if "time" in extracted.columns:
        transaction_time = pd.to_datetime(extracted["time"], errors="coerce", format="mixed")
        extracted["transaction_hour"] = transaction_time.dt.hour
        logger.info("Extracted transaction hour from 'time' column.")

    if {"total_amount", "total_purchases"}.issubset(extracted.columns):
        total_amount = pd.to_numeric(extracted["total_amount"], errors="coerce")
        total_purchases = pd.to_numeric(extracted["total_purchases"], errors="coerce")
        extracted["average_purchase_value"] = total_amount.div(total_purchases.where(total_purchases.gt(0)))
        logger.info("Extracted average purchase value from 'total_amount' and 'total_purchases' columns.")

    if {"properties_mag", "properties_sig"}.issubset(extracted.columns):
        magnitude = pd.to_numeric(extracted["properties_mag"], errors="coerce")
        significance = pd.to_numeric(extracted["properties_sig"], errors="coerce")
        extracted["earthquake_severity_score"] = magnitude * significance
        logger.info("Extracted earthquake severity score from 'properties_mag' and 'properties_sig' columns.")

    return extracted


def handle_missing_values_dask(ddf):
    """Fill missing values in Dask partitions with the pandas method."""
    logger.info("Handling missing values with Dask.")
    return ddf.map_partitions(handle_missing_values_pandas, meta=ddf._meta)


def normalize_numeric_columns_dask(ddf):
    """Standardize numeric Dask columns using global z-score statistics."""
    logger.info("Normalizing numeric columns with Dask.")
    normalized = ddf.copy()
    numeric_columns = [column for column in normalized.columns if pd.api.types.is_numeric_dtype(normalized[column].dtype) and column not in NORMALIZATION_EXCLUSIONS]

    for column in numeric_columns:
        logger.info("Normalizing column '%s' (%s).", column, normalized[column].dtype)
        values = normalized[column]
        normalized[column] = ((values - values.mean()) / values.std(ddof=0)).fillna(0)

    return normalized


def extract_features_dask(ddf):
    """Extract features in Dask partitions with the pandas method."""
    logger.info("Extracting features with Dask.")
    return ddf.map_partitions(extract_features_pandas, meta=extract_features_pandas(ddf._meta))


def optimize_with_dask(df: pd.DataFrame, npartitions: int = 4) -> pd.DataFrame:
    """Convert a pandas DataFrame to a partitioned Dask DataFrame."""
    logger.info("Optimizing DataFrame with Dask. Number of rows: %d", len(df))
    if dd is None:
        logger.warning("Dask is not installed. Continuing with pandas operations.")
        return df

    if len(df) == 0:
        return df

    ddf = dd.from_pandas(df, npartitions=npartitions)
    logger.info("Converted pandas DataFrame to Dask DataFrame with %s partitions.", ddf.npartitions)
    return ddf


def transform_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full transformation workflow for a DataFrame."""
    logger.info("Starting transformation pipeline on %d rows.", len(df))

    transformed = format_column_names(df)

    if len(df) > 20000 and dd is not None:  # Use Dask for large datasets: CSV & Database
        logger.info("Large dataset detected; using Dask optimization for processing.")
        transformed = optimize_with_dask(transformed)
        transformed = handle_missing_values_dask(transformed)
        transformed = extract_features_dask(transformed)
        transformed = normalize_numeric_columns_dask(transformed)
        transformed = transformed.compute()
    else:  # Use pandas for smaller dataset: API
        transformed = handle_missing_values_pandas(transformed)
        transformed = extract_features_pandas(transformed)
        transformed = normalize_numeric_columns_pandas(transformed)

    logger.info("Transformation pipeline complete. Final rows: %d", len(transformed))
    return transformed


def transform_dataframes(*dataframes: pd.DataFrame) -> List[pd.DataFrame]:
    """Apply the transform workflow to multiple input DataFrames."""
    return [transform_dataframe(df) for df in dataframes]
