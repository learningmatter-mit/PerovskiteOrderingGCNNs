#!/bin/bash
#SBATCH -o supercloud_job_managing/Run_jobs/Job_logs/837383/supercloud_job_runner.sh.log-%j
#SBATCH -N 1
#SBATCH -c 20
#SBATCH --gres=gpu:volta:1

source activate Perovskite_ML_Environment
python supercloud_job_managing/supercloud_job_runner.py --experiment_group_name e3nn_pretrain_1 --experiment_sigopt_id 837383 --suggestion_num 59672450