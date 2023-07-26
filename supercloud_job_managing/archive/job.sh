    #! /bin/bash
        
    #SBATCH -o Run_jobs/Job_logs/supercloud_job_runner.sh.log-%j

    #SBATCH -N 1
    #SBATCH -c 20
    #SBATCH --gres=gpu:volta:1

    module load anaconda/2021a
    source activate Perovskite_ML_Environment
    python supercloud_job_runner.py --experiment_group_name 1 --experiment_id 1 --obs_num 1