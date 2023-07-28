
import json 
import subprocess
import sigopt
import sys
sys.path.insert(0, '/home/gridsan/jdamewood/perovskites/PerovskiteOrderingGCNNs')
from training.sigopt_utils import build_sigopt_name

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
    current_jobs = current_lines - 4
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
            return line_count
        else:
            print("Error executing the command:")
            print(result.stderr)
            return -1  # Return -1 to indicate an error
    except Exception as e:
        print("An error occurred:", e)
        return -1  # Return -1 to indicate an error

def update_sigopt(experiment_group_name):

    f = open("supercloud_job_managing/experiments/" +experiment_group_name+ "/sigopt_info.json")
    sigopt_info = json.load(f)
    f.close()

    #### Search experiments 

    finished_experiment_id = None
    finished_tmp_dir = None
    model_tmp_dir = None
    reported_value = None

    for experiment_id in sigopt_info:
        for tmp_id in sigopt_info[experiment_id]["observations"]["temporary"]:
            if sigopt_info[fexperiment_id]["observations"]["temporary"][tmp_id]["status"] == "completed":


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
        return -1

    ### Report Experiment



    conn = sigopt.Connection(client_token="ERZVVPFNRCSCQIJVOVCYHVFUGZCKUDPOWZJTEXZYNOMRKQLS")

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

    if len(sigopt_info[finished_experiment_id]["observations"]["completed"]) != sigopt_num_experiments:
        return finished_experiment_id

    else:
        return None


def get_next_job(experiment_group_name,experiment_id):

    conn = sigopt.Connection(client_token="ERZVVPFNRCSCQIJVOVCYHVFUGZCKUDPOWZJTEXZYNOMRKQLS")

    suggestion = conn.experiments(experiment_id).suggestions().create()

    f = open("supercloud_job_managing/experiments/" +experiment_group_name+ "/sigopt_info.json")
    sigopt_info = json.load(f)
    f.close()

    sigot_info[experiment_id]["observations"]["temporary"][suggestion.id]["hyperparameters"] = suggestion.assignments
    sigot_info[experiment_id]["observations"]["temporary"][suggestion.id]["status"] = "running"


    f = open("supercloud_job_managing/experiments/" +experiment_group_name+ "/sigopt_info.json","w")
    json.dump(sigopt_info, f)
    f.close()

    return experiment_id, suggestion.id
