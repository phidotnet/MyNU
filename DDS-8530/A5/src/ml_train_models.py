import json
import logging
from math import sqrt
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

logger = logging.getLogger(__name__)

TARGET_COLUMN = "average_purchase_value"
NUMERIC_FEATURE_COLUMNS = ["age", "zipcode", "year", "ratings", "transaction_hour"]
CATEGORICAL_FEATURE_COLUMNS = ["income", "customer_segment", "product_category", "product_brand"]
FEATURE_COLUMNS = NUMERIC_FEATURE_COLUMNS + CATEGORICAL_FEATURE_COLUMNS
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_REGISTRY_NAMES = {
    "random_forest": "retail_average_purchase_value_random_forest",
    "gradient_boosting": "retail_average_purchase_value_gradient_boosting",
}

model_definitions = {
    "random_forest_100": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "random_forest_200": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    "gradient_boosting_100": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "gradient_boosting_200": GradientBoostingRegressor(n_estimators=200, random_state=42),
}

def evaluate_regression(model, features_test: pd.DataFrame, target_test: pd.Series) -> dict[str, float]:
    """Return MAE, RMSE, and R2 metrics for a fitted regression model."""
    predictions = model.predict(features_test)
    return {
        "mae": float(mean_absolute_error(target_test, predictions)),
        "rmse": float(sqrt(mean_squared_error(target_test, predictions))),
        "r2": float(r2_score(target_test, predictions)),
    }


def train_models(retail_transactions: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Train and save retail average-purchase-value regression models."""
    missing_columns = set(FEATURE_COLUMNS + [TARGET_COLUMN]).difference(retail_transactions.columns)
    if missing_columns:
        raise ValueError(f"Retail data is missing required columns: {sorted(missing_columns)}")

    training_data = retail_transactions[FEATURE_COLUMNS + [TARGET_COLUMN]].copy()
    training_data = training_data.dropna(subset=[TARGET_COLUMN])
    if len(training_data) < 2:
        raise ValueError("At least two rows with average_purchase_value are required for training.")

    features = training_data[FEATURE_COLUMNS]
    target = training_data[TARGET_COLUMN]
    features_train, features_test, target_train, target_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )

    metrics_by_model: dict[str, dict[str, float]] = {}
    
    # Set tracking uri
    mlflow.set_tracking_uri("sqlite:///runtime/mlflow.db")

    mlflow.set_experiment("retail_average_purchase_value")
    for model_name, estimator in model_definitions.items():
        logger.info("Training %s...", model_name)
        model = Pipeline([
            ("preprocessor", ColumnTransformer([
                ("numeric", SimpleImputer(strategy="median"), NUMERIC_FEATURE_COLUMNS),
                ("categorical", Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]), CATEGORICAL_FEATURE_COLUMNS),
            ])),
            ("regressor", estimator),
        ])

        with mlflow.start_run(run_name=model_name):
            model.fit(features_train, target_train)
            metrics = evaluate_regression(model, features_test, target_test)
            model_directory = MODELS_DIR / model_name
            model_directory.mkdir(parents=True, exist_ok=True)
            model_path = model_directory / "model.joblib"

            joblib.dump(model, model_path)
            mlflow.log_params(model.get_params())
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(str(model_path), artifact_path=model_name)
            
            # For reproducibility and version control, we log the model to MLflow and register it under the appropriate model family
            model_family = "random_forest" if model_name.startswith("random_forest") else "gradient_boosting"
            mlflow.sklearn.log_model(
                model,
                name=model_name,
                registered_model_name=MODEL_REGISTRY_NAMES[model_family],
                skops_trusted_types=["numpy.dtype"],
            )

        metrics_by_model[model_name] = metrics
        with (model_directory / "metrics.json").open("w", encoding="utf-8") as metrics_file:
            json.dump(metrics, metrics_file, indent=2)
        logger.info("=> Performance metrics: %s", metrics)

    return metrics_by_model
