# Reasoning with Sampling【WIP】

##Original Paper
### [Paper](https://arxiv.org/abs/2510.14901) | [Project Page](https://aakaran.github.io/reasoning_with_sampling/)

[![rws](teaser.png)](teaser.png)


This repo contains the official PyTorch implementation of Reasoning with Sampling.
> [**Reasoning with Sampling: Your Base Model is Smarter Than You Think**](https://arxiv.org/abs/2510.14901)<br>
> [Aayush Karan](https://aakaran.github.io/), [Yilun Du](https://yilundu.github.io/)
> <br>Harvard<br>



## Setup

Run the following script to setup environment.

```bash
git clone https://github.com/aakaran/reasoning-with-sampling.git
cd reasoning-with-sampling
conda env create -f environment.yml
conda activate cloudspace
python llm_experiments/scripts/download_eval_data.py  # fetch HumanEval/GPQA/AlpacaEval data
```


## Sampling
The llm_experiments folder contains slurm scripts to run power sampling for MATH500 (```power_samp_math.py```), whose .json is included in llm_experiments/data, as well as HumanEval (```power_samp_he.py```), GPQA Diamond (```power_samp_gpqa.py```), and AlpacaEval 2.0 (```power_samp_alpaca.py```), whose corresponding eval sets can be downloaded from their official repos. 

To run power sampling on MATH500 with 8 seeds and the eval set split across 5 shards:
```bash
sbatch llm_experiments/scripts/power_samp_math.sh
```
The output is several .csv files (based on the shard and seed number) that store the response outputs, correct answers, original prompts, etc. 

### Dynamic Sampling (optional)
To experiment with adaptive sampling during generation, set the following environment variables before running any `power_samp_*_local.sh`:

```bash
export DYNAMIC_METRIC=entropy            # or perplexity / self_confidence / none (default)
export ENTROPY_THRESHOLD=1.0             # stop when entropy <= threshold
export PERPLEXITY_THRESHOLD=3.0          # stop when perplexity <= threshold
export SELF_CONF_THRESHOLD=0.8           # stop when self-conf >= threshold
export DYNAMIC_MIN_TOKENS=64             # minimum generated tokens before checking
```

Leaving `DYNAMIC_METRIC=none` preserves the original fixed sampling behavior. When enabled, the CSVs include `dynamic_metric` and `dynamic_stop_triggered` columns for analysis.

For convenience, `llm_experiments/dynamic_runs/` contains helper scripts:

```bash
# Entropy-based stopping (defaults to math local script unless you pass another path)
bash llm_experiments/dynamic_runs/run_entropy.sh ./llm_experiments/local_scripts/power_samp_gpqa_local.sh

# Perplexity-based stopping
bash llm_experiments/dynamic_runs/run_perplexity.sh ./llm_experiments/local_scripts/power_samp_math_local.sh

# Self-confidence-based stopping
bash llm_experiments/dynamic_runs/run_self_confidence.sh ./llm_experiments/local_scripts/power_samp_he_local.sh
```

Each wrapper sets reasonable default thresholds; override by exporting the variables before running if needed.

## Evaluation
**Single-shot Reasoning**

To grade the responses for single-shot reasoning, collect the .csv files for a given seed run in a folder (e.g. ```results/qwen_math/MATH```) and pass it into ```eval_math.py```:

```bash
python llm_experiments/eval_math.py --folder=results/qwen_math/MATH
```

```eval_gpqa.py``` is similar, and for ```eval_he.py```, an additional ```--output_fname``` argument is required, as HumanEval collects all responses in a jsonl file (e.g. ```--output_fname=qwen_math_he```).

For AlpacaEval 2.0, ```eval_alpaca.py``` collects a ```--folder``` into one json file ```--output_fname```. For evaluating the json file, follow the instructions in the official repo: https://github.com/tatsu-lab/alpaca_eval


**Pass@k Performance**

For pass@k performance, collect the .csv files across seeds in a folder again (e.g. ```results/qwen_math/MATH```) and pass into ```passk_math.py```:
```bash
python llm_experiments/passk_math.py --folder=results/qwen_math/MATH
```
The output is a plot of the pass@k performance. As with single-shot reasoning, ```eval_gpqa.py``` and ```eval_he.py``` are similar, but for the latter an additional ```--output_fname``` argument is required.

## Sampling Diagnostics (Entropy/Perplexity/Self-confidence)

Each `power_samp_*.py` script now logs entropy, perplexity, and self-confidence metrics for the naive/std/mcmc trajectories into the result CSVs. After generating results for a model, you can aggregate those metrics and build a simple predictor for how many samples are required per problem:

```bash
# Aggregate metrics into analysis/sampling_predictors/sampling_metrics.csv
python llm_experiments/sampling_analysis/collect_metrics.py --folder results/qwen_math/MATH

# Train/evaluate a linear predictor for sampling counts
python llm_experiments/sampling_analysis/eval_predictor.py
```

Per-metric slices are also exported under `analysis/entropy_predictor/`, `analysis/perplexity_predictor/`, and `analysis/self_confidence_predictor/` for further experimentation.
