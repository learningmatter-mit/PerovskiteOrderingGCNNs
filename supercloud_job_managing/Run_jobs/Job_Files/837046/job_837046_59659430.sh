#!/bin/bash
#SBATCH -o supercloud_job_managing/Run_jobs/Job_logs/837046/supercloud_job_runner.sh.log-%j
#SBATCH -N 1
#SBATCH -c 20
#SBATCH --gres=gpu:volta:1

source /etc/profile
module load anaconda/2021a
source activate Perovskite_ML_Environment
python supercloud_job_runner.py --experiment_group_name test_1 --experiment_id 837046 --suggestion_num 59659430