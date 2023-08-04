#!/bin/bash
#SBATCH -o Run_jobs/Job_logs/837044/supercloud_job_runner.sh.log-%j
    #SBATCH -N 1
    #SBATCH -c 20
    #SBATCH --gres=gpu:volta:1

    module load anaconda/2021a
    source activate Perovskite_ML_Environment
    python supercloud_job_runner.py --experiment_group_name test_1 --experiment_id 837044 --suggestion_num 59659375