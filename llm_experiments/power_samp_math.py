import os
import time
import csv

from contextlib import nullcontext
from glob import glob
import json
import random
from tqdm import tqdm
import argparse

import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from datasets import Dataset, load_dataset, concatenate_datasets


import torch
import torch.nn as nn
from torch.nn import functional as F
import transformers

from grader_utils.parse_utils import parse_answer
from constants import *
from power_samp_utils import *
from metrics.generation_metrics import compute_generation_metrics, compute_metrics_from_log_probs


def _blank_metrics():
    return {"entropy": float("nan"), "perplexity": float("nan"), "self_confidence": float("nan")}


_HIDDEN_STATE_LAYER_IDX = 11  # layer-11 as used in the DPS paper (0-based; includes embedding output)

CSV_COLUMNS = [
    "question",
    "correct_answer",
    "acceptance_ratio",
    "run_mode",
    "baseline_k",
    "mcmc_k",
    "question_time_sec",
    "naive_completion",
    "naive_answer",
    "naive_entropy",
    "naive_perplexity",
    "naive_self_confidence",
    "naive_hidden_state",
    "std_completion",
    "std_answer",
    "std_entropy",
    "std_perplexity",
    "std_self_confidence",
    "mcmc_completion",
    "mcmc_answer",
    "mcmc_entropy",
    "mcmc_perplexity",
    "mcmc_self_confidence",
    "mcmc_entropy_series",
    "mcmc_perplexity_series",
    "mcmc_self_confidence_series",
    "mcmc_hidden_state",
    "std_correct",
    "naive_correct",
    "mcmc_correct",
]


def _extract_mid_hidden_state(model, token_ids, device):
    with torch.no_grad():
        outputs = model(token_ids.unsqueeze(0).to(device), output_hidden_states=True, use_cache=False)
    hidden_states = outputs.hidden_states
    if not hidden_states:
        return []
    target_idx = _HIDDEN_STATE_LAYER_IDX
    target_idx = max(0, min(target_idx, len(hidden_states) - 1))
    # This logs a single per-sample vector (last token at the chosen layer).
    # The DPS paper uses block-wise averaged layer features aggregated across trajectories.
    vec = hidden_states[target_idx][0, -1, :].detach().to("cpu")
    return vec.tolist()





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_str", action = "store", type = str, default = "results/",  dest = "save_str")
    parser.add_argument("--model", action = "store", default = "qwen", type = str, choices = ["qwen", "qwen_math", "phi", "tulu", "qwen_math_grpo", "phi_grpo"])
    parser.add_argument("--temperature", "--temp", action = "store", default = 0.25, type = float, dest = "temperature")
    parser.add_argument("--dataset", action = "store", default = "MATH", type = str)
    parser.add_argument("--cot", action = "store", type = bool, default = True)
    parser.add_argument(
        "--mcmc_steps",
        action="store",
        type=int,
        default=10,
        help="Number of MH proposals per block (not the DPS trajectory budget k).",
    )
    parser.add_argument("--device", action = "store", type = str, dest = "device", default = "cuda" if torch.cuda.is_available() else 'cpu')
    parser.add_argument("--batch_idx", action = "store", type = int, default = 0)
    parser.add_argument("--seed", action = "store", type = int, default = 0)
    parser.add_argument(
        "--run_mode",
        action="store",
        type=str,
        default="all",
        choices=["all", "baseline_only", "mcmc_only"],
        help="Run baselines+MCMC (all), only naive/std baselines, or only MCMC.")
    parser.add_argument(
        "--baseline_variant",
        action="store",
        type=str,
        default="both",
        choices=["both", "naive", "std"],
        help="Choose which baseline decoder(s) to run when run_mode includes baselines.")
    parser.add_argument(
        "--dynamic_metric",
        action="store",
        type=str,
        default="none",
        choices=["none", "entropy", "perplexity", "self_confidence"])
    parser.add_argument("--entropy_threshold", type=float, default=1.0)
    parser.add_argument("--perplexity_threshold", type=float, default=3.0)
    parser.add_argument("--self_conf_threshold", type=float, default=0.8)
    parser.add_argument("--dynamic_min_tokens", type=int, default=64)
    parser.add_argument(
        "--problem_idx",
        type=int,
        default=None,
        help="Optional 0-based index inside the batch to run only that problem.")
    args = parser.parse_args()
    if args.dynamic_metric != "none":
        raise ValueError(
            "Metric-based early stopping is disabled for MATH runs to keep results comparable."
        )

    random.seed(0)


    model = args.model
    device = args.device
    dataset_name = args.dataset
    cot = args.cot
    temp = args.temperature
    mcmc_steps = args.mcmc_steps

    temp_str = f"{temp:.6g}"
    base_output_dir = os.path.join(
        args.save_str,
        model,
        dataset_name,
        f"run_mode={args.run_mode}",
        f"baseline_variant={args.baseline_variant}",
        f"temp={temp_str}",
        f"k={mcmc_steps}",
    )
    os.makedirs(base_output_dir, exist_ok=True)


    print(model)
    print(device)
    print(mcmc_steps)
    if model == "qwen":
        model_str = "Qwen/Qwen2.5-7B"
    elif model == "qwen_math":
        model_str = "Qwen/Qwen2.5-Math-7B"
    elif model == "qwen_math_grpo":
        model_str = "stellalisy/rethink_rlvr_reproduce-ground_truth-qwen2.5_math_7b-lr5e-7-kl0.00-step150"
    elif model == "phi":
        model_str = 'microsoft/Phi-3.5-mini-instruct'
    elif model == "tulu":
        model_str = "allenai/Llama-3.1-Tulu-3-8B-DPO"

    if dataset_name == "MATH":
        json_file = 'data/MATH500.json'
        dataset = json.load(open(json_file, "r"))



    print("dataset done")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_str, trust_remote_code = True)
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(model_str, torch_dtype="auto", device_map="auto", trust_remote_code = True).to(device)
    autoreg_sampler = AutoregressiveSampler(hf_model, tokenizer, device)

    print("loaded models")

    batch_size = 100
    start = batch_size * args.batch_idx
    end = min(len(dataset), batch_size * (args.batch_idx + 1))
    batch_slice = dataset[start:end]
    if not batch_slice:
        raise ValueError(f"No samples available for batch_idx={args.batch_idx}.")
    if args.problem_idx is not None:
        if args.problem_idx < 0 or args.problem_idx >= len(batch_slice):
            raise ValueError(
                f"problem_idx={args.problem_idx} is out of range for batch_idx={args.batch_idx} "
                f"(valid range: 0-{len(batch_slice) - 1})."
            )
        iter_batch = [(args.problem_idx, batch_slice[args.problem_idx])]
    else:
        iter_batch = list(enumerate(batch_slice))

    output_fname = (
        f"{model}_{dataset_name}_run-{args.run_mode}_base-{args.baseline_variant}_"
        f"power_samp_results_{mcmc_steps}_{temp_str}_{args.batch_idx}_{args.seed}"
    )
    if args.problem_idx is not None:
        output_fname += f"_problem-{args.problem_idx}"
    output_fname += ".csv"
    output_path = os.path.join(base_output_dir, output_fname)
    with open(output_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for problem, data in tqdm(iter_batch, desc="Benchmark on MATH"):
            question_timer = time.perf_counter()
            question = data["prompt"]
            print(question)
            answer = data["answer"]

            input_text = format_prompt(question, model, tokenizer, cot)
            input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
            prefx = [idx.item() for idx in input_ids[0]]
            run_baseline = args.run_mode in ("all", "baseline_only")
            run_mcmc = args.run_mode in ("all", "mcmc_only")
            run_naive = run_baseline and args.baseline_variant in ("both", "naive")
            run_std = run_baseline and args.baseline_variant in ("both", "std")
            naive_completion = ""
            std_completion = ""
            naive_metrics = _blank_metrics()
            std_metrics = _blank_metrics()
            naive_hidden_vec = []
            if run_naive:
                naive_temp_output = hf_model.generate(
                    input_ids,
                    max_new_tokens=3072,
                    return_dict_in_generate=True,
                    output_scores=True,
                    do_sample=True,
                    temperature=temp,
                )
                naive_generated_ids = naive_temp_output[0][:, len(input_ids[0]):].squeeze().to("cpu")
                naive_completion = tokenizer.decode(naive_generated_ids, skip_special_tokens=True)
                naive_metrics = compute_generation_metrics(naive_temp_output.scores, naive_generated_ids)
                naive_ids_tensor = naive_temp_output.sequences[0].detach().clone().to(torch.long)
                naive_hidden_vec = _extract_mid_hidden_state(hf_model, naive_ids_tensor, device)
            if run_std:
                std_output = hf_model.generate(
                    input_ids,
                    max_new_tokens=3072,
                    return_dict_in_generate=True,
                    output_scores=True,
                    do_sample=True,
                )
                std_generated_ids = std_output[0][:, len(input_ids[0]):].squeeze().to("cpu")
                std_completion = tokenizer.decode(std_generated_ids, skip_special_tokens=True)
                std_metrics = compute_generation_metrics(std_output.scores, std_generated_ids)
            acceptance_ratio = float("nan")
            mcmc_completion = ""
            mcmc_metrics = _blank_metrics()
            mcmc_power_samp_ids = torch.tensor([], dtype=torch.long)
            mcmc_hidden_vec = []
            mcmc_entropy_series = []
            mcmc_perplexity_series = []
            mcmc_self_conf_series = []
            if run_mcmc:
                mcmc_power_samp_output, log_probs_norm, _, acceptance_ratio = mcmc_power_samp(
                    autoreg_sampler,
                    prefx,
                    temp,
                    mcmc_steps,
                    max_new_tokens=3072,
                )
                mcmc_power_samp_ids = torch.tensor([mcmc_power_samp_output], dtype=torch.long, device=device).squeeze().to("cpu")
                mcmc_completion = tokenizer.decode(mcmc_power_samp_ids, skip_special_tokens=True)
                mcmc_metrics = compute_metrics_from_log_probs(log_probs_norm)
                mcmc_entropy_series = mcmc_metrics.get("entropy_series", [])
                mcmc_perplexity_series = mcmc_metrics.get("perplexity_series", [])
                mcmc_self_conf_series = mcmc_metrics.get("self_confidence_series", [])
                mcmc_ids_tensor = torch.tensor(mcmc_power_samp_output, dtype=torch.long)
                mcmc_hidden_vec = _extract_mid_hidden_state(hf_model, mcmc_ids_tensor, device)

            naive_answer = parse_answer(naive_completion) if run_baseline else ""
            std_answer = parse_answer(std_completion) if run_baseline else ""
            mcmc_answer = parse_answer(mcmc_completion) if run_mcmc else ""
            
            if run_naive:
                print(naive_answer)
            if run_std:
                print(std_answer)
            if run_mcmc:
                print(mcmc_answer)
            print(question)
            print(answer)
            print(f'Acceptance: {acceptance_ratio}')


            elapsed = time.perf_counter() - question_timer

            record = {col: "" for col in CSV_COLUMNS}
            record.update({
                "question": question,
                "correct_answer": answer,
                "acceptance_ratio": acceptance_ratio,
                "run_mode": args.run_mode,
                "baseline_k": 1 if run_baseline else 0,
                "mcmc_k": mcmc_steps if run_mcmc else 0,
                "question_time_sec": elapsed,
            })
            if run_naive:
                record.update({
                    "naive_completion": naive_completion,
                    "naive_answer": naive_answer,
                    "naive_entropy": naive_metrics["entropy"],
                    "naive_perplexity": naive_metrics["perplexity"],
                    "naive_self_confidence": naive_metrics["self_confidence"],
                    "naive_hidden_state": json.dumps(naive_hidden_vec) if naive_hidden_vec else "",
                })
            if run_std:
                record.update({
                    "std_completion": std_completion,
                    "std_answer": std_answer,
                    "std_entropy": std_metrics["entropy"],
                    "std_perplexity": std_metrics["perplexity"],
                    "std_self_confidence": std_metrics["self_confidence"],
                })
            if run_mcmc:
                record.update({
                    "mcmc_completion": mcmc_completion,
                    "mcmc_answer": mcmc_answer,
                    "mcmc_entropy": mcmc_metrics["entropy"],
                    "mcmc_perplexity": mcmc_metrics["perplexity"],
                    "mcmc_self_confidence": mcmc_metrics["self_confidence"],
                    "mcmc_entropy_series": json.dumps(mcmc_entropy_series),
                    "mcmc_perplexity_series": json.dumps(mcmc_perplexity_series),
                    "mcmc_self_confidence_series": json.dumps(mcmc_self_conf_series),
                    "mcmc_hidden_state": json.dumps(mcmc_hidden_vec) if mcmc_hidden_vec else "",
                })

            writer.writerow(record)












        
