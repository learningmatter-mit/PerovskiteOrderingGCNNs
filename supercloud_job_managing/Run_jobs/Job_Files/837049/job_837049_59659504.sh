#!/bin/bash
#SBATCH -o supercloud_job_managing/Run_jobs/Job_logs/837049/supercloud_job_runner.sh.log-%j
#SBATCH -N 1
#SBATCH -c 20
#SBATCH --gres=gpu:volta:1

module load anaconda/2021a
source activate Perovskite_ML_Environment
python supercloud_job_managing/supercloud_job_runner.py --experiment_group_name test_1 --experiment_sigopt_id 837049 --suggestion_num 59659504