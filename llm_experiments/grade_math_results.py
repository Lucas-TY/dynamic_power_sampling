#!/usr/bin/env python
"""
Scan the results/ directory for MATH benchmark CSV files, grade each answer column,
and write correctness columns back into the CSV. Skips files that already contain
the correctness columns.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

from grader_utils.math_grader import grade_answer


ANSWER_COLUMNS = {
    "std_answer": "std_correct",
    "naive_answer": "naive_correct",
    "mcmc_answer": "mcmc_correct",
}


def safe_grade(answer: str, correct: str) -> int:
    try:
        return int(grade_answer(answer, correct))
    except Exception:
        return 0


def find_math_csvs(root: Path) -> List[Path]:
    """Return all CSV files that appear to be MATH benchmark outputs."""
    candidates: List[Path] = []
    for csv_path in root.rglob("*.csv"):
        name = csv_path.name.lower()
        if "_math_" in name or name.startswith("math_"):
            candidates.append(csv_path)
    return sorted(candidates)


def detect_correct_answer_column(df: pd.DataFrame) -> str:
    for col in ("correct_answer", "Correct Answer"):
        if col in df.columns:
            return col
    raise ValueError("CSV does not contain a recognized correct_answer column.")


def needs_grading(df: pd.DataFrame, force: bool) -> bool:
    if force:
        return True
    return any(col not in df.columns for col in ANSWER_COLUMNS.values())


def add_correctness_columns(df: pd.DataFrame, overwrite: bool = False) -> pd.DataFrame:
    correct_col = detect_correct_answer_column(df)
    correct_series = df[correct_col].fillna("")
    updates: Dict[str, List[int]] = {}
    for answer_col, result_col in ANSWER_COLUMNS.items():
        if answer_col not in df.columns:
            continue
        if not overwrite and result_col in df.columns:
            continue
        answers = df[answer_col].fillna("")
        updates[result_col] = [
            safe_grade(ans, correct)
            for ans, correct in zip(answers, correct_series)
        ]
    for column, values in updates.items():
        if column in df.columns:
            df[column] = values
        else:
            df[column] = values
    return df


def grade_file(csv_path: Path, force: bool) -> bool:
    try:
        df = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        print(f"[skip-empty] {csv_path}")
        return False
    if not needs_grading(df, force):
        return False
    df = add_correctness_columns(df, overwrite=force)
    df.to_csv(csv_path, index=False)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("results"),
        help="Root directory containing result CSV files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute correctness columns even if they already exist.",
    )
    args = parser.parse_args()

    csv_files = find_math_csvs(args.root)
    if not csv_files:
        print(f"No MATH CSV files found under {args.root}")
        return

    graded = 0
    for csv_path in csv_files:
        did_update = grade_file(csv_path, force=args.force)
        status = "updated" if did_update else "skipped"
        print(f"[{status}] {csv_path}")
        if did_update:
            graded += 1
    print(f"Processed {len(csv_files)} files; updated {graded}.")


if __name__ == "__main__":
    main()
