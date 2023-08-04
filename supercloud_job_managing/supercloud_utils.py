import time

import json 
import subprocess
import sigopt
import sys
sys.path.insert(0, '/home/gridsan/jdamewood/perovskites/PerovskiteOrderingGCNNs')
from training.sigopt_utils import build_sigopt_name
import shutil
import os
import torch


def check_if_experiment_ongoing(name):

    f = open("supercloud_job_managing/experiments/" +name+ "/sigopt_info.json")
    experimental_group_sigopt = json.load(f)
    f.close()

    for experiment_id in experimental_group_sigopt:
        sigopt_num_experiments = experimental_group_sigopt[experiment_id]["settings"]["sigopt_settings"]["obs_budget"]
        if len(experimental_group_sigopt[experiment_id]["observations"]["completed"]) != sigopt_num_experiments:
            return True

    return False

def check_for_job_space():
    current_lines = get_command_output_line_count("LLstat -p xeon-g6-volta")

    current_jobs = current_lines - 3

    print("Number of current gpus jobs is: " + str(current_jobs) + "\n")
    if current_jobs < 8:
        return True
    return False


def get_command_output_line_count(command):
    try:
        # Run the Bash command and capture the output
        result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Check if the command executed successfully
        if result.returncode == 0:
            # Split the output by lines and count the number of lines
            line_count = len(result.stdout.strip().split('\n'))
            #print(line_count)
            return line_count
        else:
            print("Error executing the command:")
            print(result.stderr)
            return -1  # Return -1 to indicate an error
    except Exception as e:
        print("An error occurred:", e)
        return -1  # Return -1 to indicate an error

def update_sigopt(experiment_group_name):


    conn = sigopt.Connection(client_token="ERZVVPFNRCSCQIJVOVCYHVFUGZCKUDPOWZJTEXZYNOMRKQLS")


    f = open("supercloud_job_managing/experiments/" +experiment_group_name+ "/sigopt_info.json")
    sigopt_info = json.load(f)
    f.close()

    #### Search experiments 

    finished_experiment_id = None
    finished_tmp_dir = None
    model_tmp_dir = None
    reported_value = None

    ### Clear Broken Jobs

    for experiment_id in sigopt_info:
        curr_time = time.time()
        for tmp_id in sigopt_info[experiment_id]["observations"]["temporary"]:
            if float(curr_time - sigopt_info[experiment_id]["observations"]["temporary"][tmp_id]["start_time"])/3600 > 12.0:
                sigopt_info[experiment_id]["observations"]["temporary"].pop(tmp_id)
                suggestion_deleted = conn.experiments(experiment_id).suggestions(tmp_id).delete()


    for experiment_id in sigopt_info:
        if len(sigopt_info[experiment_id]["observations"]["temporary"]) == 0 and len(sigopt_info[experiment_id]["observations"]["completed"]) == 0:
            return experiment_id

        for tmp_id in sigopt_info[experiment_id]["observations"]["temporary"]:
            if sigopt_info[experiment_id]["observations"]["temporary"][tmp_id]["status"] == "completed":


                finished_experiment_id = experiment_id
                finished_tmp_id = tmp_id

                settings = sigopt_info[finished_experiment_id]["settings"]
                model_type = settings["model_type"]
                sigopt_name = build_sigopt_name(settings["data_name"],settings["target_prop"],settings["struct_type"],settings["interpolation"],settings["model_type"])

                model_tmp_dir = './saved_models/'+ model_type + '/' + sigopt_name + '/' + str(finished_experiment_id) + '/' + str(tmp_id) + '_tmp' + str(0)

                f = open(model_tmp_dir + "/training_results.json")
                training_results = json.load(f)
                f.close()
                reported_value = training_results["validation_loss"]

    if finished_experiment_id == None:
        for experiment_id in sigopt_info:
            sigopt_num_experiments = sigopt_info[experiment_id]["settings"]["sigopt_settings"]["obs_budget"]
            if len(sigopt_info[experiment_id]["observations"]["temporary"]) + len(sigopt_info[experiment_id]["observations"]["completed"]) < sigopt_num_experiments and len(sigopt_info[experiment_id]["observations"]["temporary"]) < sigopt_info[experiment_id]["settings"]["sigopt_settings"]["parallel_band"]:
                return experiment_id

        return -1

    ### Report Experiment


    conn.experiments(finished_experiment_id).observations().create(
            suggestion=finished_tmp_id,
            value=reported_value,
        )

    sigopt_info[finished_experiment_id]["observations"]["completed"][finished_tmp_id] = sigopt_info[finished_experiment_id]["observations"]["temporary"][finished_tmp_id]
    sigopt_info[finished_experiment_id]["observations"]["temporary"].pop(finished_tmp_id)

    f = open("supercloud_job_managing/experiments/" + experiment_group_name + "/sigopt_info.json","w")
    json.dump(sigopt_info, f)
    f.close()

    ### Move Files

    experiment = conn.experiments(finished_experiment_id).fetch()
    observation_id = experiment.progress.observation_count - 1

    settings = sigopt_info[finished_experiment_id]["settings"]

    model_type = settings["model_type"]

    sigopt_name = build_sigopt_name(settings["data_name"],settings["target_prop"],settings["struct_type"],settings["interpolation"],settings["model_type"])

    model_save_dir = './saved_models/'+ model_type + '/' + sigopt_name + '/' + str(finished_experiment_id) + '/observ_' + str(observation_id)

    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    ### Copy contents of tmp file
    possible_file_names = ["best_model", "best_model.pth.tar", "best_model.torch",
                            "final_model.torch","final_model","final_model.pth.tar",
                            "log_human_read.csv","checkpoints/checkpoint-100.pth.tar","training_results.json"]
    for file_name in possible_file_names:
        if os.path.isfile(model_tmp_dir + "/" + file_name):
            if file_name == "checkpoints/checkpoint-100.pth.tar":
                shutil.move(model_tmp_dir + "/" + file_name, model_save_dir + "/" + "checkpoint-100.pth.tar")
            else:
                shutil.move(model_tmp_dir + "/" + file_name, model_save_dir + "/" + file_name)
        
    ### Empty tmp file
    shutil.rmtree(model_tmp_dir)

    torch.cuda.empty_cache()

    sigopt_num_experiments = sigopt_info[finished_experiment_id]["settings"]["sigopt_settings"]["obs_budget"]

    if len(sigopt_info[finished_experiment_id]["observations"]["temporary"]) + len(sigopt_info[finished_experiment_id]["observations"]["completed"]) < sigopt_num_experiments:
        return finished_experiment_id

    else:
        return None


def get_next_job(experiment_group_name,experiment_id):

    conn = sigopt.Connection(client_token="ERZVVPFNRCSCQIJVOVCYHVFUGZCKUDPOWZJTEXZYNOMRKQLS")

    suggestion = conn.experiments(experiment_id).suggestions().create()

    f = open("supercloud_job_managing/experiments/" +experiment_group_name+ "/sigopt_info.json")
    sigopt_info = json.load(f)
    f.close()

    sigopt_info[experiment_id]["observations"]["temporary"][suggestion.id]["hyperparameters"] = suggestion.assignments
    sigopt_info[experiment_id]["observations"]["temporary"][suggestion.id]["status"] = "running"
    sigopt_info[experiment_id]["observations"]["temporary"][suggestion.id]["start_time"] = time.time()


    f = open("supercloud_job_managing/experiments/" +experiment_group_name+ "/sigopt_info.json","w")
    json.dump(sigopt_info, f)
    f.close()

    return experiment_id, suggestion.id
