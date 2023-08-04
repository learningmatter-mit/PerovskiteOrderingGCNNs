import json
import argparse
import time
import os
import sys
sys.path.insert(0, '/home/gridsan/jdamewood/perovskites/PerovskiteOrderingGCNNs')
from supercloud_utils import *
from write_job_file import write_job_file
import subprocess
from create_sigopt_experiments import create_sigopt_experiments_from_Group


def manage_experiment(name):


    while check_if_experiment_ongoing(name):

        time.sleep(60)

        print("Checking For Job Space \n")
        if check_for_job_space():
            print("Found Job Space \n")
            experiment_id = update_sigopt(name)

            if experiment_id != None and experiment_id != -1:

                print("Submitting a job \n")

                experiment_id, suggestion_num = get_next_job(name,experiment_id)

                ### Write Job

                write_job_file(name,experiment_id,suggestion_num)

                ### Run Job

                job_name = 'job_' + str(experiment_id) + '_' + str(suggestion_num) + '.sh'


                subprocess.check_call(['chmod','u+x',"supercloud_job_managing/"+job_name])

                subprocess.check_call(['sbatch',"supercloud_job_managing/"+job_name])

                ### Move Job

                os.rename("supercloud_job_managing/"+job_name, "supercloud_job_managing/Run_jobs/Job_Files/" + str(experiment_id) + "/" + job_name) 
                



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get Name for Experiments')

    parser.add_argument('--name', default = "", type=str, metavar='name',
                        help="the path of the experiment Group")

    parser.add_argument('--create', default = "False", type=str, metavar='creation',
                        help="whether to creatre a new sigopt_info file")


    args = parser.parse_args()
    file_path = args.name
    create = args.create
    if create == "True":
        print("Creating Sigopt Experiments")
        create_sigopt_experiments_from_Group(file_path)

    manage_experiment(file_path)

