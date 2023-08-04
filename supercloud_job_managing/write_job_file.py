#! /usr/bin/env python

import argparse


def write_job_file(experiment_group_name,experiment_id,suggestion_num):

    f =  open('supercloud_job_managing/job_' + str(experiment_id) + '_' + str(suggestion_num) + '.sh', 'w')

    f.write('#!/bin/bash\n')
    f.write('#SBATCH -o supercloud_job_managing/Run_jobs/Job_logs/' + str(experiment_id) + '/supercloud_job_runner.sh.log-%j'
    '''
#SBATCH -N 1
#SBATCH -c 20
#SBATCH --gres=gpu:volta:1

source activate Perovskite_ML_Environment\n'''
    )

    f.write('python supercloud_job_managing/supercloud_job_runner.py --experiment_group_name ' + experiment_group_name + ' --experiment_sigopt_id ' + experiment_id + ' --suggestion_num ' + str(suggestion_num))

    f.close()
    
