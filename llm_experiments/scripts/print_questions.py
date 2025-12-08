#!/usr/bin/env python
"""
Print the question and correct_answer columns for every row in a CSV.
Usage:
    python print_questions.py path/to/results.csv
"""

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path, help="Path to the CSV file.")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)
    if "question" not in df.columns or "correct_answer" not in df.columns:
        raise ValueError("CSV must contain 'question' and 'correct_answer' columns.")

    for idx, row in df.iterrows():
        question = row["question"]
        answer = row["correct_answer"]
        print(f"[{idx}] question: {question}")
        print(f"[{idx}] correct_answer: {answer}")
        print("-" * 80)


if __name__ == "__main__":
    main()
