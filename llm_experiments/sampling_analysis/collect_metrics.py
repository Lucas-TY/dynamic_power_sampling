#!/usr/bin/env python
"""
Aggregate sampling metrics (entropy, perplexity, self-confidence) from result CSV files.
This script groups rows by question, infers how many samples were required to reach a correct answer,
and writes merged features for downstream prediction tasks.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


RESULT_PATTERN = re.compile(
    r".*_power_samp_results_(?P<mcmc_steps>\d+)_(?P<temp>[0-9.]+)_(?P<batch>\d+)_(?P<seed>\d+)\.csv"
)


def parse_metadata(path: Path) -> Dict[str, float]:
    match = RESULT_PATTERN.match(path.name)
    if not match:
        return {"mcmc_steps": np.nan, "temperature": np.nan, "batch_idx": np.nan, "seed": np.nan}
    groups = match.groupdict()
    return {
        "mcmc_steps": int(groups["mcmc_steps"]),
        "temperature": float(groups["temp"]),
        "batch_idx": int(groups["batch"]),
        "seed": int(groups["seed"]),
    }


def load_results(folder: Path) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for csv_path in sorted(folder.rglob("*.csv")):
        df = pd.read_csv(csv_path)
        for key, value in parse_metadata(csv_path).items():
            df[key] = value
        df["source_file"] = csv_path.name
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No CSV files found under {folder}")
    return pd.concat(frames, ignore_index=True)


def samples_to_success(rows: pd.DataFrame) -> Tuple[Optional[int], int]:
    sorted_rows = rows.sort_values(["batch_idx", "seed"]).reset_index(drop=True)
    for idx, row in sorted_rows.iterrows():
        if str(row.get("mcmc_answer", "")).strip() == str(row.get("correct_answer", row.get("Correct Answer", ""))).strip():
            return idx + 1, len(sorted_rows)
    return None, len(sorted_rows)


def build_feature_row(rows: pd.DataFrame) -> Dict[str, float]:
    sorted_rows = rows.sort_values(["batch_idx", "seed"]).reset_index(drop=True)
    first = sorted_rows.iloc[0]
    label, total = samples_to_success(rows)
    feature = {
        "question": first.get("question", first.get("instruction", "")),
        "correct_answer": first.get("correct_answer", first.get("Correct Answer", "")),
        "naive_entropy": first.get("naive_entropy", np.nan),
        "naive_perplexity": first.get("naive_perplexity", np.nan),
        "naive_self_confidence": first.get("naive_self_confidence", np.nan),
        "std_entropy": first.get("std_entropy", np.nan),
        "std_perplexity": first.get("std_perplexity", np.nan),
        "std_self_confidence": first.get("std_self_confidence", np.nan),
        "mcmc_entropy": first.get("mcmc_entropy", np.nan),
        "mcmc_perplexity": first.get("mcmc_perplexity", np.nan),
        "mcmc_self_confidence": first.get("mcmc_self_confidence", np.nan),
        "samples_to_success": label if label is not None else np.nan,
        "total_samples": total,
        "dynamic_metric": first.get("dynamic_metric", "none"),
        "dynamic_stop_triggered": first.get("dynamic_stop_triggered", False),
    }
    return feature


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--folder",
        type=Path,
        required=True,
        help="Folder containing power_samp result CSV files (e.g., results/qwen_math/MATH)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis/sampling_predictors/sampling_metrics.csv"),
        help="Output CSV path for aggregated metrics.",
    )
    args = parser.parse_args()

    df = load_results(args.folder)
    grouped = df.groupby("question" if "question" in df.columns else "instruction")
    features = [build_feature_row(group.copy()) for _, group in grouped]
    feature_df = pd.DataFrame(features)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_csv(args.output, index=False)
    print(f"[write] Saved aggregated metrics to {args.output}")

    entropy_cols = ["question", "naive_entropy", "std_entropy", "mcmc_entropy", "samples_to_success"]
    perplex_cols = ["question", "naive_perplexity", "std_perplexity", "mcmc_perplexity", "samples_to_success"]
    confidence_cols = ["question", "naive_self_confidence", "std_self_confidence", "mcmc_self_confidence", "samples_to_success"]

    feature_df[entropy_cols].to_csv("analysis/entropy_predictor/metrics.csv", index=False)
    feature_df[perplex_cols].to_csv("analysis/perplexity_predictor/metrics.csv", index=False)
    feature_df[confidence_cols].to_csv("analysis/self_confidence_predictor/metrics.csv", index=False)
    print("[write] Exported per-metric slices to analysis/entropy_predictor|perplexity_predictor|self_confidence_predictor.")


if __name__ == "__main__":
    main()
