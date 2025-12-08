RUN_MODE=mcmc_only MCMC_STEPS=4 bash llm_experiments/local_scripts/power_samp_math_local.sh 2 5 --save_str results/math_mcmc_k4/MATH
RUN_MODE=mcmc_only MCMC_STEPS=6 bash llm_experiments/local_scripts/power_samp_math_local.sh 2 5 --save_str results/math_mcmc_k6/MATH
RUN_MODE=mcmc_only MCMC_STEPS=8 bash llm_experiments/local_scripts/power_samp_math_local.sh 2 5 --save_str results/math_mcmc_k8/MATH
RUN_MODE=mcmc_only MCMC_STEPS=10 bash llm_experiments/local_scripts/power_samp_math_local.sh 2 5 --save_str results/math_mcmc_k10/MATH