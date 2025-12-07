#!/usr/bin/env python
"""
Download evaluation datasets required by the power_samp scripts.

Usage:
  python download_eval_data.py [--data-dir llm_experiments/data]
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
import json

from huggingface_hub import hf_hub_download

DATASETS = {
    "ALPACA": {
        "type": "url",
        "url": "https://huggingface.co/datasets/tatsu-lab/alpaca_eval/resolve/main/alpaca_farm_human_crossannotations.json",
        "target": "ALPACA.json",
        "description": "AlpacaEval 2.0 evaluation prompts (cross-annotations file)",
    },
    "GPQA": {
        "type": "hf_hub",
        "repo_id": "Idavidrein/gpqa",
        "filename": "gpqa_diamond.json",
        "repo_type": "dataset",
        "target": "GPQA.jsonl",
        "description": "GPQA Diamond split",
        "convert_to_jsonl": True,
    },
    "HUMANEVAL": {
        "type": "datasets",
        "dataset_name": "openai_humaneval",
        "split": "test",
        "target": "HumanEval.jsonl",
        "description": "OpenAI HumanEval benchmark",
    },
}


def download_dataset(name: str, entry: dict, data_dir: Path) -> None:
    target_path = data_dir / entry["target"]
    if target_path.exists():
        print(f"[skip] {name}: {target_path} already exists.")
        return

    print(f"[download] {name}: fetching {entry['description']}")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    source_type = entry.get("type", "hf_hub")

    if source_type == "url":
        import requests

        print(f"  -> downloading from {entry['url']}")
        resp = requests.get(entry["url"], timeout=60)
        resp.raise_for_status()
        target_path.write_bytes(resp.content)
    elif source_type == "hf_hub":
        cached_file = hf_hub_download(
            repo_id=entry["repo_id"],
            filename=entry["filename"],
            repo_type=entry.get("repo_type", "dataset"),
            resume_download=True,
        )
        if entry.get("convert_to_jsonl"):
            with open(cached_file, "r", encoding="utf-8") as src, target_path.open(
                "w", encoding="utf-8"
            ) as dst:
                data = json.load(src)
                for record in data:
                    json.dump(record, dst, ensure_ascii=False)
                    dst.write("\n")
        else:
            shutil.copy(cached_file, target_path)
    elif source_type == "datasets":
        from datasets import load_dataset

        split = entry.get("split", "train")
        dataset = load_dataset(entry["dataset_name"], split=split)
        with target_path.open("w", encoding="utf-8") as f:
            for record in dataset:
                json.dump(record, f, ensure_ascii=False)
                f.write("\n")
    else:
        raise ValueError(f"Unsupported dataset type for {name}: {source_type}")

    print(f"[done] Saved to {target_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data",
        help="Directory to store datasets (default: llm_experiments/data)",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        choices=sorted(DATASETS.keys()),
        help="Specific dataset names to download (default: all).",
    )
    args = parser.parse_args()

    selected = DATASETS
    if args.datasets:
        selected = {name: DATASETS[name] for name in args.datasets}

    for name, entry in selected.items():
        download_dataset(name, entry, args.data_dir)


if __name__ == "__main__":
    main()
