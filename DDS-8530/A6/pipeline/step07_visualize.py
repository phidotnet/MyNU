from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

if __package__:
    from pipeline.common import *
else:
    from common import *

def plot_missingness_heatmap(products: pd.DataFrame, output_dir: Path) -> Path:
    """Heatmap of missing cells, labeled with the % of observed values per column."""
    observed_pct = (1 - products.isna().mean()) * 100
    labels = [f"{column} ({observed_pct[column]:.1f}%)" for column in products.columns]
    total_observed_pct = (1 - products.isna().values.mean()) * 100
    figure, axis = plt.subplots(figsize=(12, 6))
    sns.heatmap(products.isna(), cbar=False, yticklabels=False, cmap="magma", ax=axis)
    axis.set_xticks([index + 0.5 for index in range(len(products.columns))])
    axis.set_xticklabels(labels, rotation=45, ha="right")
    axis.set(title=f"Missingness Heatmap ({total_observed_pct:.1f}% Observed Overall)")
    figure.tight_layout()
    path = output_dir / "viz_missingness_heatmap.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def _numeric_stats(valid: pd.Series) -> dict[str, float]:
    n = len(valid)
    if n == 0:
        return {"n": 0, "min": float("nan"), "max": float("nan"), "mean": float("nan"), "median": float("nan"), "kurtosis": float("nan"), "outliers": 0, "outlier_pct": 0.0}
    quartile_1, quartile_3 = valid.quantile(0.25), valid.quantile(0.75)
    iqr = quartile_3 - quartile_1
    lower, upper = quartile_1 - 1.5 * iqr, quartile_3 + 1.5 * iqr
    outliers = valid[(valid < lower) | (valid > upper)]
    return {
        "n": n,
        "min": valid.min(),
        "max": valid.max(),
        "mean": valid.mean(),
        "median": valid.median(),
        "kurtosis": valid.kurt(),
        "outliers": len(outliers),
        "outlier_pct": len(outliers) / n * 100,
    }


def _annotate_stats(axis, stats: dict[str, float]) -> None:
    text = (
        f"n = {stats['n']:,}\n"
        f"min = {stats['min']:.2f}\n"
        f"max = {stats['max']:.2f}\n"
        f"mean = {stats['mean']:.2f}\n"
        f"median = {stats['median']:.2f}\n"
        f"kurtosis (K) = {stats['kurtosis']:.2f}\n"
        f"outliers = {stats['outliers']:,} ({stats['outlier_pct']:.1f}%)"
    )
    axis.text(0.98, 0.98, text, transform=axis.transAxes, ha="right", va="top", bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.8"})


def plot_numeric_distribution(products: pd.DataFrame, column: str, output_dir: Path) -> Path:
    """Linear/log distribution pair for a numeric column, annotated with summary stats."""
    values = pd.to_numeric(products[column], errors="coerce")
    valid = values.dropna()
    valid_log = valid[valid > 0]
    stats = _numeric_stats(valid)

    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    if valid.empty:
        axes[0].text(0.5, 0.5, "No data", ha="center", va="center")
    else:
        sns.histplot(valid, bins=30, color="#4c72b0", ax=axes[0])
    axes[0].set(title=f"{column} (Linear Scale)", xlabel=column, ylabel="Number of Products")

    if valid_log.empty:
        axes[1].text(0.5, 0.5, "No positive values", ha="center", va="center")
    else:
        sns.histplot(valid_log, bins=30, log_scale=True, color="#4c72b0", ax=axes[1])
    axes[1].set(title=f"{column} (Log Scale)", xlabel=f"{column} (log scale)", ylabel="Number of Products")

    for axis in axes:
        _annotate_stats(axis, stats)
    figure.suptitle(f"Distribution of {column}")
    figure.tight_layout()
    path = output_dir / f"viz_distribution_{column}.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def plot_categorical_distribution(products: pd.DataFrame, column: str, output_dir: Path) -> Path:
    """Top-value bar chart for a non-numeric column, annotated with n/missing/distinct."""
    values = products[column]
    non_null = values.dropna()
    distinct = non_null.nunique()
    top_counts = non_null.value_counts().head(15)

    figure, axis = plt.subplots(figsize=(10, 5))
    sns.barplot(x=top_counts.values, y=top_counts.index.astype(str), ax=axis, color="#55a868")
    axis.set(title=f"Top Values for {column}", xlabel="Count", ylabel=column)
    text = f"n = {len(non_null):,}\nmissing = {len(values) - len(non_null):,}\ndistinct = {distinct:,}"
    axis.text(0.98, 0.02, text, transform=axis.transAxes, ha="right", va="bottom", bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.8"})
    figure.tight_layout()
    path = output_dir / f"viz_distribution_{column}.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def plot_all_distributions(products: pd.DataFrame, output_dir: Path, method_logger) -> list[Path]:
    paths = []
    for column in products.columns:
        if column == "_id":
            continue
        phase_started = perf_counter()
        if pd.api.types.is_numeric_dtype(products[column]):
            path = plot_numeric_distribution(products, column, output_dir)
        else:
            path = plot_categorical_distribution(products, column, output_dir)
        paths.append(path)
        method_logger.info("Completed distribution column=%s elapsed_seconds=%.4f", column, perf_counter() - phase_started)
    return paths


@log_duration
def run(input_path: Path = DATASET_CLEAN_FILE) -> list[Path]:
    method_logger = get_method_logger(run)

    # Load the cleaned camera records used for both visualizations.
    phase_started = perf_counter()
    method_logger.info("START load_products")
    if not input_path.exists():
        raise FileNotFoundError("Run pipeline.ingest_clean before visualization.")
    output_dir = ensure_output_dir()
    products = pd.read_csv(input_path)
    method_logger.info("END load_products records=%s elapsed_seconds=%.4f", len(products), perf_counter() - phase_started)

    sns.set_theme(style="whitegrid")
    paths = []

    # Create and save the missing-value heatmap.
    phase_started = perf_counter()
    method_logger.info("START missingness_heatmap")
    paths.append(plot_missingness_heatmap(products, output_dir))
    method_logger.info("END missingness_heatmap elapsed_seconds=%.4f", perf_counter() - phase_started)

    # Create and save one distribution chart per column, except _id.
    phase_started = perf_counter()
    method_logger.info("START distributions")
    paths.extend(plot_all_distributions(products, output_dir, method_logger))
    method_logger.info("END distributions elapsed_seconds=%.4f", perf_counter() - phase_started)

    method_logger.info("outputs=%s", [str(path) for path in paths])
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate data quality visualizations.")
    parser.add_argument("--input", type=Path, default=DATASET_CLEAN_FILE)
    arguments = parser.parse_args()
    run(arguments.input)