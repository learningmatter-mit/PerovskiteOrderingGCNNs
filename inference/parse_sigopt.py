import sigopt
import numpy as np
import json
from run_sigopt_experiment import build_sigopt_name

def parse_sigopt(prop,model_settings,num_of_models = 3):

    conn = sigopt.Connection(client_token="ERZVVPFNRCSCQIJVOVCYHVFUGZCKUDPOWZJTEXZYNOMRKQLS")

    experiment_id = get_experiment_id(model_settings)

    all_observations = conn.experiments(experiment_id).observations().fetch()
    all_losses = []
    all_assignments = []

    for observation in all_observations.data:
        all_losses.append(observation.values[0].value)
        all_assignments.append(observation.assignments)

    best_idx = np.argpartition(np.asarray(all_losses),num_of_models)
    best_assignments = all_assignments[best_idx]
    sigopt_name = build_sigopt_name(prop,model_settings["relaxed"],model_settings["interpolation"],model_settings["model_type"])

    for i in range(len(best_idx)):

        new_directory = "./saved_models/best_models/" + sigopt_name + "/" + "best_model_" + str(i)
        old_directory = "./saved_models/" + model_settings["model_type"] + "/" + sigopt_name + "/" + "observ_" + str(best_idx[i])

        if not os.path.exists(new_dir):
            os.makedirs(new_dir) 

        ### save assignments to new_dictionary
        with open(new_directory + '/assignments.json', 'w') as f:
            json.dump(best_assignments[i], f)

        ### copy and move best_model to new_dictionary
        possible_file_names = ["best_model", "best_model.pth.tar", "best_model.torch"]
        for file_name in possible_file_names:
            if os.path.isfile(old_directory + "/" + file_name):
                shutil.copy(old_directory + "/" + file_name, new_directory + "/" + file_name)
    

    

def get_experiment_id(model_params):
    if model_params["model_type"] == "CGCNN" and model_params["interpolation"] == False and model_params["relaxed"] == False:
        return 595923
    elif model_params["model_type"] == "CGCNN" and model_params["interpolation"] == False and model_params["relaxed"] == True:
        return 595922
    elif model_params["model_type"] == "CGCNN" and model_params["interpolation"] == True and model_params["relaxed"] == False:
        return 595934
    elif model_params["model_type"] == "CGCNN" and model_params["interpolation"] == True and model_params["relaxed"] == True:
        return 595935
    elif model_params["model_type"] == "Painn" and model_params["interpolation"] == True and model_params["relaxed"] == False:
        return 595924
    elif model_params["model_type"] == "Painn" and model_params["interpolation"] == True and model_params["relaxed"] == True:
        return 595925
    elif model_params["model_type"] == "e3nn" and model_params["interpolation"] == True and model_params["relaxed"] == False:
        return 595926
    elif model_params["model_type"] == "e3nn" and model_params["interpolation"] == True and model_params["relaxed"] == True:
        return 595927
    elif model_params["model_type"] == "e3nn_contrastive" and model_params["interpolation"] == True and model_params["relaxed"] == False:
        return 595937
    elif model_params["model_type"] == "e3nn_contrastive" and model_params["interpolation"] == True and model_params["relaxed"] == True:
        return 595936

    else:
        raise ValueError('These model parameters have not been studied')


if __name__ == '__main__':

    prop = "dft_e_hull"

    model_settings = {
        "model_type": "CGCNN",
        "relaxed": True,
        "interpolation": True,
    }
    parse_sigopt(prop,model_settings,num_of_models = 3)
    model_settings = {
        "model_type": "CGCNN",
        "relaxed": True,
        "interpolation": False,
    }
    parse_sigopt(prop,model_settings,num_of_models = 3)
    model_settings = {
        "model_type": "CGCNN",
        "relaxed": False,
        "interpolation": True,
    }
    parse_sigopt(prop,model_settings,num_of_models = 3)
    model_settings = {
        "model_type": "CGCNN",
        "relaxed": False,
        "interpolation": False,
    }
    parse_sigopt(prop,model_settings,num_of_models = 3)

    model_settings = {
        "model_type": "Painn",
        "relaxed": True,
        "interpolation": True,
    }
    parse_sigopt(prop,model_settings,num_of_models = 3)
    model_settings = {
        "model_type": "Painn",
        "relaxed": False,
        "interpolation": True,
    }
    parse_sigopt(prop,model_settings,num_of_models = 3)

    model_settings = {
        "model_type": "e3nn",
        "relaxed": True,
        "interpolation": True,
    }
    parse_sigopt(prop,model_settings,num_of_models = 3)
    model_settings = {
        "model_type": "e3nn",
        "relaxed": False,
        "interpolation": True,
    }
    parse_sigopt(prop,model_settings,num_of_models = 3)

    model_settings = {
        "model_type": "e3nn_contrastive",
        "relaxed": True,
        "interpolation": True,
    }
    parse_sigopt(prop,model_settings,num_of_models = 3)
    model_settings = {
        "model_type": "e3nn_contrastive",
        "relaxed": False,
        "interpolation": True,
    }
    parse_sigopt(prop,model_settings,num_of_models = 3)