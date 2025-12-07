#!/usr/bin/env python
"""Extract hidden states and correctness labels from math results CSVs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import pandas as pd

from grader_utils.math_grader import grade_answer


def safe_grade(pred: str, truth: str) -> int:
    try:
        return int(grade_answer(pred, truth))
    except Exception:
        return 0


def export_hidden_states(folder: Path, output: Path) -> None:
    records: List[dict] = []
    for csv_path in sorted(folder.glob("*.csv")):
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            question = row.get("question", "")
            correct = row.get("correct_answer", "")
            run_mode = row.get("run_mode", "all")

            hidden_json = row.get("naive_hidden_state", "")
            if hidden_json:
                hidden = json.loads(hidden_json)
                label = safe_grade(row.get("naive_answer", ""), correct)
                records.append(
                    {
                        "question": question,
                        "hidden_state": hidden,
                        "k": row.get("baseline_k", 1),
                        "correct": label,
                        "mode": "baseline",
                        "source_file": csv_path.name,
                        "run_mode": run_mode,
                    }
                )

            hidden_json = row.get("mcmc_hidden_state", "")
            if hidden_json:
                hidden = json.loads(hidden_json)
                label = safe_grade(row.get("mcmc_answer", ""), correct)
                records.append(
                    {
                        "question": question,
                        "hidden_state": hidden,
                        "k": row.get("mcmc_k", 10),
                        "correct": label,
                        "mode": "mcmc",
                        "source_file": csv_path.name,
                        "run_mode": run_mode,
                    }
                )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "
")
    print(f"[export] wrote {len(records)} entries to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export_hidden_states(args.folder, args.output)


if __name__ == "__main__":
    main()
