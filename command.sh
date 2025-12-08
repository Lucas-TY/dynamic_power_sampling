# ---------- Fixed sampling runs ----------
# MATH (baselines separated by decoder, then multiple MCMC settings)
RUN_MODE=baseline_only BASELINE_VARIANT=naive MCMC_STEPS=0 bash llm_experiments/local_scripts/power_samp_math_local.sh 2 5 --save_str results/math_baseline_naive/MATH_k0
BASELINE_VARIANT=std RUN_MODE=baseline_only bash llm_experiments/local_scripts/power_samp_math_local.sh 2 5 --save_str results/math_baseline_std/MATH_k0

RUN_MODE=mcmc_only MCMC_STEPS=2 bash llm_experiments/local_scripts/power_samp_math_local.sh 2 5 --save_str results/math_mcmc_k2/MATH
RUN_MODE=mcmc_only MCMC_STEPS=4 bash llm_experiments/local_scripts/power_samp_math_local.sh 2 5 --save_str results/math_mcmc_k4/MATH
RUN_MODE=mcmc_only MCMC_STEPS=6 bash llm_experiments/local_scripts/power_samp_math_local.sh 2 5 --save_str results/math_mcmc_k6/MATH
RUN_MODE=mcmc_only MCMC_STEPS=8 bash llm_experiments/local_scripts/power_samp_math_local.sh 2 5 --save_str results/math_mcmc_k8/MATH
RUN_MODE=mcmc_only MCMC_STEPS=10 bash llm_experiments/local_scripts/power_samp_math_local.sh 2 5 --save_str results/math_mcmc_k10/MATH
# HumanEval
bash llm_experiments/local_scripts/power_samp_he_local.sh 0 7 --save_str results/fixed_runs/HE
# AlpacaEval
bash llm_experiments/local_scripts/power_samp_alpaca_local.sh 5 0 --save_str results/fixed_runs/ALPACA
# GPQA
bash llm_experiments/local_scripts/power_samp_gpqa_local.sh 1 3 --save_str results/fixed_runs/GPQA

# ---------- Dynamic sampling runs (uncomment the ones you need) ----------
# Entropy-based
RUN_MODE=mcmc_only bash llm_experiments/dynamic_runs/run_entropy.sh ./llm_experiments/local_scripts/power_samp_math_local.sh 2 5 --save_str results/math_dynamic_entropy/MATH
bash llm_experiments/dynamic_runs/run_entropy.sh ./llm_experiments/local_scripts/power_samp_he_local.sh 0 7 --save_str results/dynamic_entropy/HE
bash llm_experiments/dynamic_runs/run_entropy.sh ./llm_experiments/local_scripts/power_samp_alpaca_local.sh 5 0 --save_str results/dynamic_entropy/ALPACA
bash llm_experiments/dynamic_runs/run_entropy.sh ./llm_experiments/local_scripts/power_samp_gpqa_local.sh 1 3 --save_str results/dynamic_entropy/GPQA

# Perplexity-based
RUN_MODE=mcmc_only bash llm_experiments/dynamic_runs/run_perplexity.sh ./llm_experiments/local_scripts/power_samp_math_local.sh 2 5 --save_str results/math_dynamic_perplexity/MATH
bash llm_experiments/dynamic_runs/run_perplexity.sh ./llm_experiments/local_scripts/power_samp_he_local.sh 0 7 --save_str results/dynamic_perplexity/HE
bash llm_experiments/dynamic_runs/run_perplexity.sh ./llm_experiments/local_scripts/power_samp_alpaca_local.sh 5 0 --save_str results/dynamic_perplexity/ALPACA
bash llm_experiments/dynamic_runs/run_perplexity.sh ./llm_experiments/local_scripts/power_samp_gpqa_local.sh 1 3 --save_str results/dynamic_perplexity/GPQA

# Self-confidence-based
RUN_MODE=mcmc_only bash llm_experiments/dynamic_runs/run_self_confidence.sh ./llm_experiments/local_scripts/power_samp_math_local.sh 2 5 --save_str results/math_dynamic_confidence/MATH
bash llm_experiments/dynamic_runs/run_self_confidence.sh ./llm_experiments/local_scripts/power_samp_he_local.sh 0 7 --save_str results/dynamic_confidence/HE
bash llm_experiments/dynamic_runs/run_self_confidence.sh ./llm_experiments/local_scripts/power_samp_alpaca_local.sh 5 0 --save_str results/dynamic_confidence/ALPACA
bash llm_experiments/dynamic_runs/run_self_confidence.sh ./llm_experiments/local_scripts/power_samp_gpqa_local.sh 1 3 --save_str results/dynamic_confidence/GPQA

# ---------- Evaluation ----------
# Fixed sampling results
python llm_experiments/eval_math.py --folder=results/math_baseline_naive/MATH_k1
python llm_experiments/eval_math.py --folder=results/math_baseline_std/MATH_k1
python llm_experiments/eval_math.py --folder=results/math_mcmc_k2/MATH
python llm_experiments/eval_math.py --folder=results/math_mcmc_k4/MATH
python llm_experiments/eval_math.py --folder=results/math_mcmc_k6/MATH
python llm_experiments/eval_math.py --folder=results/math_mcmc_k8/MATH
python llm_experiments/eval_math.py --folder=results/math_mcmc_k10/MATH
python llm_experiments/eval_he.py --folder=results/fixed_runs/HE
python llm_experiments/eval_alpaca.py --folder=results/fixed_runs/ALPACA
python llm_experiments/eval_gpqa.py --folder=results/fixed_runs/GPQA

# Dynamic (entropy)
python llm_experiments/eval_math.py --folder=results/math_dynamic_entropy/MATH
python llm_experiments/eval_he.py --folder=results/dynamic_entropy/HE
python llm_experiments/eval_alpaca.py --folder=results/dynamic_entropy/ALPACA
python llm_experiments/eval_gpqa.py --folder=results/dynamic_entropy/GPQA

# Dynamic (perplexity)
python llm_experiments/eval_math.py --folder=results/math_dynamic_perplexity/MATH
python llm_experiments/eval_he.py --folder=results/dynamic_perplexity/HE
python llm_experiments/eval_alpaca.py --folder=results/dynamic_perplexity/ALPACA
python llm_experiments/eval_gpqa.py --folder=results/dynamic_perplexity/GPQA

# Dynamic (self-confidence)
python llm_experiments/eval_math.py --folder=results/math_dynamic_confidence/MATH
python llm_experiments/eval_he.py --folder=results/dynamic_confidence/HE
python llm_experiments/eval_alpaca.py --folder=results/dynamic_confidence/ALPACA
python llm_experiments/eval_gpqa.py --folder=results/dynamic_confidence/GPQA

# Hidden state exports for training
python llm_experiments/sampling_analysis/export_hidden_states.py --folder=results/math_baseline_naive/MATH_k1 --output=analysis/hidden_states/math_k1_naive.jsonl
python llm_experiments/sampling_analysis/export_hidden_states.py --folder=results/math_baseline_std/MATH_k1 --output=analysis/hidden_states/math_k1_std.jsonl
python llm_experiments/sampling_analysis/export_hidden_states.py --folder=results/math_mcmc_k2/MATH --output=analysis/hidden_states/math_k2.jsonl
python llm_experiments/sampling_analysis/export_hidden_states.py --folder=results/math_mcmc_k4/MATH --output=analysis/hidden_states/math_k4.jsonl
python llm_experiments/sampling_analysis/export_hidden_states.py --folder=results/math_mcmc_k6/MATH --output=analysis/hidden_states/math_k6.jsonl
python llm_experiments/sampling_analysis/export_hidden_states.py --folder=results/math_mcmc_k8/MATH --output=analysis/hidden_states/math_k8.jsonl
python llm_experiments/sampling_analysis/export_hidden_states.py --folder=results/math_mcmc_k10/MATH --output=analysis/hidden_states/math_k10.jsonl

python llm_experiments/sampling_analysis/export_hidden_states.py --folder=results/math_dynamic_entropy/MATH --output=analysis/hidden_states/math_entropy.jsonl
python llm_experiments/sampling_analysis/export_hidden_states.py --folder=results/math_dynamic_perplexity/MATH --output=analysis/hidden_states/math_perplexity.jsonl
python llm_experiments/sampling_analysis/export_hidden_states.py --folder=results/math_dynamic_confidence/MATH --output=analysis/hidden_states/math_confidence.jsonl
