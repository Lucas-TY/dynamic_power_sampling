import os

from contextlib import nullcontext
from glob import glob
import json
import random
from tqdm import tqdm
import argparse

import pandas as pd
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
from metrics.dynamic_controller import DynamicStopController


def _blank_metrics():
    return {"entropy": float("nan"), "perplexity": float("nan"), "self_confidence": float("nan")}


def mcmc_power_samp_alp(p : AutoregressiveSampler, context, temp, mcmc_steps, max_new_tokens, block_num=16, stop_controller=None):
    c = len(context)
    print(f'Temp: {temp}')
    gen = []
    if context is not None:
        gen = context.copy()
    log_probs_norm = []
    log_probs_unnorm = []


    print(max_new_tokens)
    assert max_new_tokens % block_num == 0
    jump_size = int(max_new_tokens // block_num)
    print(jump_size)
    attempts = 0
    acceptances = 0


    for _ in tqdm(range(block_num)):
        gen, lp_norm, lp_unnorm = naive_temp(p, gen, 0.5, seq_len=jump_size+len(gen))
        log_probs_norm.extend(lp_norm)
        log_probs_unnorm.extend(lp_unnorm)

        for _ in tqdm(range(mcmc_steps)):
            attempts+=1
            t = len(gen)
            idx = random.randint(c, t-1)
            # llm query takes the burden of time
            prop, log_prob_prop, target_log_prob_prop = naive_temp(p, gen[:idx], 0.5, seq_len=t)
            s = len(prop)
            assert(len(log_prob_prop) == s - idx)
            assert(len(target_log_prob_prop) == s - idx)
            log_prob_cur = log_probs_norm.copy()[idx-c:s-c]
            target_log_prob_cur = log_probs_unnorm.copy()[idx-c:s-c]
            log_r = sum(target_log_prob_prop) + sum(log_prob_cur) - sum(target_log_prob_cur) - sum(log_prob_prop)

            if np.random.rand() < np.exp(log_r):
                acceptances+=1
                gen = prop.copy()
                log_probs_norm[idx-c:] = log_prob_prop.copy()
                log_probs_unnorm[idx-c:] = target_log_prob_prop.copy()

                del prop
                del log_prob_prop
                del target_log_prob_cur

        if stop_controller is not None and stop_controller(log_probs_norm):
            acceptance_ratio = acceptances/attempts if attempts else 0.0
            return gen, log_probs_norm, log_probs_unnorm, acceptance_ratio

        if p.tokenizer.eos_token_id in gen:
            eos_idx = gen.index(p.tokenizer.eos_token_id)
            gen = gen[:eos_idx + 1]
            log_probs_norm = log_probs_norm[:eos_idx + 1]
            log_probs_unnorm = log_probs_unnorm[:eos_idx + 1]
            acceptance_ratio = acceptances/attempts if attempts else 0.0
            return gen, log_probs_norm, log_probs_unnorm, acceptance_ratio

    acceptance_ratio = acceptances/attempts if attempts else 0.0
    return gen, log_probs_norm, log_probs_unnorm, acceptance_ratio



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_str", action = "store", type = str, default = "results/",  dest = "save_str")
    parser.add_argument("--model", action = "store", default = "qwen", type = str, choices = ["qwen", "qwen_math", "phi", "tulu", "qwen_math_grpo", "phi_grpo"])
    parser.add_argument("--temperature", "--temp", action = "store", default = 0.25, type = float, dest = "temperature")
    parser.add_argument("--dataset", action = "store", default = "ALPACA", type = str)
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
    parser.add_argument("--dynamic_metric", action="store", type=str, default="none",
                        choices=["none", "entropy", "perplexity", "self_confidence"])
    parser.add_argument("--entropy_threshold", type=float, default=1.0)
    parser.add_argument("--perplexity_threshold", type=float, default=3.0)
    parser.add_argument("--self_conf_threshold", type=float, default=0.8)
    parser.add_argument("--dynamic_min_tokens", type=int, default=64)
    args = parser.parse_args()

    random.seed(0)


    model = args.model
    device = args.device
    dataset_name = args.dataset
    cot = args.cot
    temp = args.temperature
    mcmc_steps = args.mcmc_steps

    save_str = os.path.join(args.save_str, model)
    os.makedirs(save_str, exist_ok=True)


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

    if dataset_name == "ALPACA":
        json_file = 'data/ALPACA.json'
        dataset = json.load(open(json_file, "r"))
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    N = len(dataset)



    print("dataset done")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_str, trust_remote_code = True)
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(model_str, torch_dtype="auto", device_map="auto", trust_remote_code = True).to(device)
    autoreg_sampler = AutoregressiveSampler(hf_model, tokenizer, device)

    print("loaded models")
    results = []

    start = 58*args.batch_idx
    end = min(58*(args.batch_idx+1), N)

    for problem, data in tqdm(enumerate(dataset[start:end]), desc = "Benchmark on ALPACA"):
        source = data["dataset"]
        instruction = data["instruction"]
        generator = model
        input_text = instruction
        print(input_text)
        
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
        prefx = [idx.item() for idx in input_ids[0]]

        naive_temp_output = hf_model.generate(input_ids, max_new_tokens=3072, 
                                return_dict_in_generate=True, output_scores=True, do_sample = True, temperature = temp)
        
        print(tokenizer.decode(naive_temp_output[0][:, len(input_ids[0]):].squeeze().to("cpu"), skip_special_tokens=True))
        print("naive done")
        
        
        std_output = hf_model.generate(input_ids, max_new_tokens=3072, 
                                return_dict_in_generate=True, output_scores=True, do_sample = True)
        
        print(tokenizer.decode(std_output[0][:, len(input_ids[0]):].squeeze().to("cpu"), skip_special_tokens=True))
        print("std done")

        stop_controller = None
        if args.dynamic_metric != "none":
            stop_controller = DynamicStopController(
                metric=args.dynamic_metric,
                entropy_threshold=args.entropy_threshold,
                perplexity_threshold=args.perplexity_threshold,
                self_conf_threshold=args.self_conf_threshold,
                min_tokens=args.dynamic_min_tokens,
            )
        mcmc_power_samp_output, log_probs_norm, _, acceptance_ratio = mcmc_power_samp_alp(
            autoreg_sampler,
            prefx,
            temp,
            mcmc_steps,
            max_new_tokens=3072,
            stop_controller=stop_controller,
        )

        print(len(std_output))
        print(len(naive_temp_output))
        print(len(mcmc_power_samp_output))
        print(tokenizer.decode(torch.tensor([mcmc_power_samp_output], dtype=torch.long, device=device).squeeze().to("cpu"), skip_special_tokens=True))
        print("mcmc done")

        naive_generated_ids = naive_temp_output[0][:, len(input_ids[0]):].squeeze().to("cpu")
        std_generated_ids = std_output[0][:, len(input_ids[0]):].squeeze().to("cpu")
        mcmc_power_samp_ids = torch.tensor([mcmc_power_samp_output], dtype=torch.long, device=device).squeeze().to("cpu")

        naive_completion = tokenizer.decode(naive_generated_ids, skip_special_tokens=True)
        std_completion = tokenizer.decode(std_generated_ids, skip_special_tokens=True)
        mcmc_completion = tokenizer.decode(mcmc_power_samp_ids, skip_special_tokens=True)

        if args.dynamic_metric != "none":
            naive_metrics = compute_generation_metrics(naive_temp_output.scores, naive_generated_ids)
            std_metrics = compute_generation_metrics(std_output.scores, std_generated_ids)
            mcmc_metrics = compute_metrics_from_log_probs(log_probs_norm)
        else:
            naive_metrics = _blank_metrics()
            std_metrics = _blank_metrics()
            mcmc_metrics = _blank_metrics()


        results.append({
            "dataset": source,
            "instruction": instruction,
            "naive_completion": naive_completion,
            "std_completion": std_completion,
            "mcmc_completion": mcmc_completion,
            "naive_entropy": naive_metrics["entropy"],
            "naive_perplexity": naive_metrics["perplexity"],
            "naive_self_confidence": naive_metrics["self_confidence"],
            "std_entropy": std_metrics["entropy"],
            "std_perplexity": std_metrics["perplexity"],
            "std_self_confidence": std_metrics["self_confidence"],
            "mcmc_entropy": mcmc_metrics["entropy"],
            "mcmc_perplexity": mcmc_metrics["perplexity"],
            "mcmc_self_confidence": mcmc_metrics["self_confidence"],
            "acceptance_ratio": acceptance_ratio,
            "dynamic_metric": args.dynamic_metric,
            "dynamic_stop_triggered": stop_controller.triggered if stop_controller else False,
        })

    
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(save_str, model+"_alpaca_base_power_samp_results_" + str(mcmc_steps) + "_" + str(temp) + "_" + str(args.batch_idx)  + "_" + str(args.seed) + ".csv"), index=False)
    
