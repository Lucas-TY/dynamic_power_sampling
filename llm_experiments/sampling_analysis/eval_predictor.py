#!/usr/bin/env python
"""
Train a simple regression model to predict sampling counts from entropy/perplexity/self-confidence features.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics-file",
        type=Path,
        default=Path("analysis/sampling_predictors/sampling_metrics.csv"),
        help="Aggregated metrics CSV produced by collect_metrics.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis/sampling_predictors/predictions.csv"),
        help="Path to store predictions for inspection.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.metrics_file)
    df = df.dropna(subset=["samples_to_success"])
    if df.empty:
        raise ValueError("No rows with samples_to_success labels were found.")

    feature_cols = [
        "naive_entropy",
        "naive_perplexity",
        "naive_self_confidence",
        "std_entropy",
        "std_perplexity",
        "std_self_confidence",
        "mcmc_entropy",
        "mcmc_perplexity",
        "mcmc_self_confidence",
    ]
    X = df[feature_cols].fillna(df[feature_cols].mean())
    y = df["samples_to_success"].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    mae = mean_absolute_error(y_test, preds)
    corr, _ = spearmanr(y_test, preds)

    print(f"Eval set size: {len(y_test)}")
    print(f"RMSE: {rmse:.3f}")
    print(f"MAE: {mae:.3f}")
    print(f"Spearman correlation: {corr:.3f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame({"ground_truth": y_test, "prediction": preds})
    out_df.to_csv(args.output, index=False)
    print(f"[write] Saved prediction samples to {args.output}")


if __name__ == "__main__":
    main()
