#! /usr/bin/env python

import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NN training parameters')
    parser.add_argument('--experiment_group_name', type=str,help="the name of the experiment group", required=True,)
    parser.add_argument('--experiment_id', type=str,help="the id of the particular experiment", required=True,)
    parser.add_argument('--obs_num',type=str,help="the observation to run", required=True,)

    args = parser.parse_args()
    experiment_group_name = args.experiment_group_name
    experiment_id = args.experiment_id
    obs_num = args.obs_num


    f =  open('job.sh', 'w')

    f.write('''\
    #! /bin/bash
        
    #SBATCH -o Run_jobs/Job_logs/supercloud_job_runner.sh.log-%j

    #SBATCH -N 1
    #SBATCH -c 20
    #SBATCH --gres=gpu:volta:1

    module load anaconda/2021a
    source activate Perovskite_ML_Environment
    '''
    )

    f.write('python supercloud_job_runner.py --experiment_group_name ' + experiment_group_name + ' --experiment_id ' + experiment_id + ' --obs_num ' + obs_num)

    f.close()
    
