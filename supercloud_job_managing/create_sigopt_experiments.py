import sigopt
import json
import argparse
from step_2_run_sigopt_experiment import create_sigopt_experiment

def create_sigopt_experiments_from_Group(file_name):

    conn = sigopt.Connection(client_token="ERZVVPFNRCSCQIJVOVCYHVFUGZCKUDPOWZJTEXZYNOMRKQLS")

    #### Load Json

    f = open("supercloud_job_managing/experiments/" +file_name+ "/settings.json")
    experimental_group_settings = json.load(f)
    f.close()

    sigopt_dict = {}
    ### For Experiment in Json
    for experiment_idx in experimental_group_settings:

        data_name = experimental_settings[experiment_idx]["data_name"]
        target_prop = experimental_settings[experiment_idx]["target_prop"]
        struct_type = experimental_settings[experiment_idx]["struct_type"]
        interpolation = experimental_settings[experiment_idx]["interpolation"]
        model_type = experimental_settings[experiment_idx]["model_type"]
        sigopt_settings = experimental_settings[experiment_idx]["sigopt_settings"]

        sigopt_experiment = create_sigopt_experiment(data_name,target_prop,struct_type,interpolation,model_type,sigopt_settings,conn)

        experiment_dict = {}
        experiment_dict["sigopt_id"] = sigot_experiment.id
        experiment_dict["observations"] = {}

        sigopt_dict[experiment_idx] = experiment_dict

    with open("supercloud_job_managing/experiments/" +file_name+ "/sigopt_info.json") as outfile:
        json.dump(sigopt_dict, outfile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get Settings for Experiments')

    parser.add_argument('--name', default = "", type=str, metavar='name',
                        help="the path of the experiment Group")

    args = parser.parse_args()
    file_path = args.name
    create_sigopt_experiments_from_Group(file_path)