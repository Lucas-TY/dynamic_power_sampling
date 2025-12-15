# Dynamic Power Sampling for LLM Reasoning

This repo contains code and experiment utilities accompanying **Dynamic Power Sampling for LLM Reasoning**.

[![rws](teaser.png)](teaser.png)

## What this is based on

This project builds on the **power sampling** method and its public implementation from:
- [Reasoning with Sampling: Your Base Model is Smarter Than You Think](https://arxiv.org/abs/2510.14901) (Karan & Du, 2025)
- Project page: https://aakaran.github.io/reasoning_with_sampling/

The DPS paper keeps the base model, power \(\alpha\), proposal distribution, and MH schedule unchanged, and only changes **how the trajectory budget is chosen per problem**.

## Paper

### Dynamic Power Sampling (this work)
- `Dynamic_Power_Sampling.pdf`
- Authors: Tianyu Wu, Yuzhen Chen

### Power Sampling (base method)
- [Reasoning with Sampling: Your Base Model is Smarter Than You Think](https://arxiv.org/abs/2510.14901) (Karan & Du, 2025)
- Project page: https://aakaran.github.io/reasoning_with_sampling/

## Overview

Power sampling targets a *sharpened* distribution over reasoning traces:
\[
\pi_\alpha(x\mid q) \propto p_\theta(x\mid q)^\alpha,\ \alpha>1,
\]
and approximately samples from it using Metropolis–Hastings (MH) updates in token space.

The original setup uses a fixed number of trajectories `k` for every problem (e.g. `k=8`), which wastes compute on easy instances.
**Dynamic Power Sampling (DPS)** keeps the base model and MH schedule fixed, but chooses the trajectory budget per problem:
- Start with `k=2`
- At a few checkpoints, a small controller predicts whether the current budget is sufficient
- If not, increase `k` by 2 (up to `k=8`)

## What’s implemented in this repo

- Fixed-\(k\) power sampling runners for MATH500/HumanEval/GPQA/AlpacaEval under `llm_experiments/`.
- Logging of DPS-relevant scalar diagnostics (entropy/perplexity/self-confidence, MH acceptance ratio, runtime) into CSVs.
- Lightweight hidden-state extraction utilities (note: the paper uses block-wise averaged layer-11 features aggregated across trajectories; this repo currently logs a single vector per sample for analysis).
- Optional: metric-based early stopping for a single MH chain (separate from DPS’s dynamic-`k` scheduling).

## Controllers (as described in the paper)

All controllers share the same scalar features (computed at each checkpoint):
- `mcmc_perplexity`: average token-level perplexity under the base model
- `mcmc_entropy`: average token-level entropy of the next-token distribution
- `mcmc_self_confidence`: average top-1 probability
- `acceptance_ratio`: fraction of accepted MH proposals
- `question_time_sec`: elapsed wall-clock time
- `current_k`: current trajectory budget

Controller variants:
- **DPS-GBDT**: gradient-boosted decision tree on the scalar features
- **DPS-MLP-H**: MLP on scalar + layer-11 hidden-state features
- **DPS-MLP-HQ**: MLP on scalar + hidden-state + question embedding

Decision rule (paper notation):
\[
\hat{p}_{\text{inc}} = 1-\hat{p}_{\text{enough}},\ \ \text{increase } k \text{ if } \hat{p}_{\text{inc}} > \tau \text{ and } k<8.
\]

## Main Result (MATH500, Qwen3-7B; from the paper)

| Method | Accuracy (%) | Time (s / question) |
| --- | ---: | ---: |
| Fixed-`k` power sampling (`k=8`) | 69.7 | 263.2 |
| DPS-GBDT | 68.7 | 73.2 |
| DPS-MLP-H | 69.7 | 104.2 |
| DPS-MLP-HQ | **70.7** | 87.8 |

Most MATH500 questions are solved with `k=2`, and the controller only escalates on a small fraction of harder instances.

## Setup

```bash
conda env create -f environment.yml
conda activate cloudspace
python llm_experiments/scripts/download_eval_data.py  # fetch HumanEval/GPQA/AlpacaEval data
```

## Running Experiments

The `llm_experiments/` folder contains runnable scripts for power sampling on:
- MATH500 (`llm_experiments/power_samp_math.py`, dataset included in `llm_experiments/data/MATH500.json`)
- HumanEval (`llm_experiments/power_samp_he.py`)
- GPQA Diamond (`llm_experiments/power_samp_gpqa.py`)
- AlpacaEval 2.0 (`llm_experiments/power_samp_alpaca.py`)

To run MATH500 on a Slurm cluster (5 shards × 8 seeds):
```bash
sbatch llm_experiments/scripts/power_samp_math.sh
```

## Evaluation

Single-shot grading:
```bash
python llm_experiments/eval_math.py --folder=results/qwen_math/MATH
```

Pass@k plots:
```bash
python llm_experiments/passk_math.py --folder=results/qwen_math/MATH
```

## Metrics Logging (for DPS features)

This repo logs entropy/perplexity/self-confidence (and MH acceptance ratios) into the result CSVs to support controller training.
You can aggregate those metrics and fit a lightweight predictor:

```bash
# Aggregate metrics into analysis/sampling_predictors/sampling_metrics.csv
python llm_experiments/sampling_analysis/collect_metrics.py --folder results/qwen_math/MATH

# Train/evaluate a linear predictor for sampling counts
python llm_experiments/sampling_analysis/eval_predictor.py
```

Per-metric slices are also exported under `analysis/entropy_predictor/`, `analysis/perplexity_predictor/`, and `analysis/self_confidence_predictor/` for further experimentation.

## Optional: Metric-Based Early Stopping (separate from DPS)

This repo also contains an experimental *early stopping* controller that halts MH sampling when entropy/perplexity/self-confidence crosses a threshold.
This is **not** the same as the paper’s DPS controller (which schedules the trajectory budget `k`).

Wrappers live in `llm_experiments/dynamic_runs/` (for local scripts):
```bash
bash llm_experiments/dynamic_runs/run_entropy.sh ./llm_experiments/local_scripts/power_samp_gpqa_local.sh
```
