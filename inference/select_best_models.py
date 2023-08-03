import os
import shutil
import json
import sigopt
import math
import torch
import pandas as pd
from processing.dataloader.dataloader import get_dataloader
from processing.utils import filter_data_by_properties,select_structures
from training.evaluate import evaluate_model
from processing.interpolation.Interpolation import *
from training.loss import contrastive_loss
from training.sigopt_utils import build_sigopt_name
from processing.create_model.create_model import create_model


def get_experiment_id(model_params, target_prop):
    if target_prop == "dft_e_hull":
        if model_params["model_type"] == "CGCNN" and model_params["interpolation"] == False and model_params["struct_type"] == 'unrelaxed':
            return 836636
        elif model_params["model_type"] == "CGCNN" and model_params["interpolation"] == False and model_params["struct_type"] == 'relaxed':
            return 836637
        elif model_params["model_type"] == "CGCNN" and model_params["interpolation"] == False and model_params["struct_type"] == 'M3Gnet_relaxed':
            return 836638
        elif model_params["model_type"] == "e3nn" and model_params["interpolation"] == False and model_params["struct_type"] == 'unrelaxed':
            return 836629
        elif model_params["model_type"] == "e3nn" and model_params["interpolation"] == False and model_params["struct_type"] == 'relaxed':
            return 836630
        elif model_params["model_type"] == "e3nn" and model_params["interpolation"] == False and model_params["struct_type"] == 'M3Gnet_relaxed':
            return 836631
        else:
            raise ValueError('These model parameters have not been studied')
    
    elif target_prop == "Op_band_center":    
        if model_params["model_type"] == "CGCNN" and model_params["interpolation"] == False and model_params["struct_type"] == 'unrelaxed':
            return 836639
        elif model_params["model_type"] == "CGCNN" and model_params["interpolation"] == False and model_params["struct_type"] == 'relaxed':
            return 836640
        elif model_params["model_type"] == "CGCNN" and model_params["interpolation"] == False and model_params["struct_type"] == 'M3Gnet_relaxed':
            return 836632
        elif model_params["model_type"] == "e3nn" and model_params["interpolation"] == False and model_params["struct_type"] == 'unrelaxed':
            return 836633
        elif model_params["model_type"] == "e3nn" and model_params["interpolation"] == False and model_params["struct_type"] == 'relaxed':
            return 836634
        elif model_params["model_type"] == "e3nn" and model_params["interpolation"] == False and model_params["struct_type"] == 'M3Gnet_relaxed':
            return 836635
        else:
            raise ValueError('These model parameters have not been studied')

    else:
        raise ValueError('This target property has not been studied')

def load_model(gpu_num, train_loader, target_prop, model_params, folder_idx, job_idx):
    device_name = "cuda:" + str(gpu_num)
    device = torch.device(device_name)
    
    sigopt_name = build_sigopt_name("data/", target_prop, model_params["struct_type"], model_params["interpolation"], model_params["model_type"])
    exp_id = get_experiment_id(model_params, target_prop)
    directory = "./saved_models/" + model_params["model_type"] + "/"+ sigopt_name + "/" +str(exp_id)+"/" + "observ_" + str(folder_idx)
    
    conn = sigopt.Connection(client_token="ERZVVPFNRCSCQIJVOVCYHVFUGZCKUDPOWZJTEXZYNOMRKQLS")
    all_observations = conn.experiments(exp_id).observations().fetch()
    assignments = all_observations.data[job_idx].assignments

    if model_params["model_type"] == "Painn":
        model = torch.load(directory + "/best_model", map_location=device)
        normalizer = None
    else:
        model, normalizer = create_model(model_params["model_type"], train_loader, assignments)
        model.to(device)
        model.load_state_dict(torch.load(directory + "/best_model.torch", map_location=device)['state'])
    
    return model, normalizer


def reverify_sigopt_models(model_params, gpu_num, target_prop):
    device_name = "cuda:" + str(gpu_num)
    device = torch.device(device_name)
    torch.cuda.set_device(device)
    
    interpolation = model_params["interpolation"]
    model_type = model_params["model_type"]    
    
    training_data = pd.read_json('data/' + 'training_set.json')
    validation_data = pd.read_json('data/' + 'validation_set.json')
    edge_data = pd.read_json('data/' + 'edge_dataset.json')    

    print("Loaded data")

    if not interpolation:
        training_data = pd.concat((training_data,edge_data))

    data = [training_data, validation_data]
    processed_data = []

    for dataset in data:
        dataset = filter_data_by_properties(dataset,target_prop)
        dataset = select_structures(dataset,model_params["struct_type"])

        if interpolation:
            dataset = apply_interpolation(dataset,target_prop)

        processed_data.append(dataset)

    print("Completed data processing")
    
    train_data = processed_data[0]
    validation_data = processed_data[1]

    train_loader = get_dataloader(train_data,target_prop,model_type,1,interpolation)
    val_loader = get_dataloader(validation_data,target_prop,model_type,1,interpolation)       

    reverify_sigopt_models_results = pd.DataFrame(columns=['sigopt_loss', 'reverified_loss'])

    conn = sigopt.Connection(client_token="ERZVVPFNRCSCQIJVOVCYHVFUGZCKUDPOWZJTEXZYNOMRKQLS")
    exp_id = get_experiment_id(model_params, target_prop)
    all_observations = conn.experiments(exp_id).observations().fetch()
    
    for folder_idx in range(len(all_observations.data)):        
        job_idx = len(all_observations.data) - folder_idx - 1
        print('Reverifying sigopt model #' + str(folder_idx))

        sigopt_loss = all_observations.data[job_idx].value
        
        hyperparameters = all_observations.data[job_idx].assignments
        sigopt_name = build_sigopt_name("data/", target_prop, model_params["struct_type"], model_params["interpolation"], model_params["model_type"])
        directory = "./saved_models/" + model_params["model_type"] + "/"+ sigopt_name + "/" +str(exp_id)+"/" + "observ_" + str(folder_idx)
        with open(directory + '/hyperparameters.json', 'w') as file:
            json.dump(hyperparameters, file)

        model, normalizer = load_model(gpu_num, train_loader, target_prop, model_params, folder_idx, job_idx)
        
        if "contrastive" in model_type:
            loss_fn = contrastive_loss 
        else:
            loss_fn = torch.nn.L1Loss()

        _, _, best_loss = evaluate_model(model, normalizer, model_type, val_loader, loss_fn, gpu_num) 

        if model_type == "Painn":
            reverified_loss = best_loss
        else:
            reverified_loss = best_loss[0]

        new_row = pd.DataFrame([[sigopt_loss, reverified_loss]], columns=['sigopt_loss', 'reverified_loss'])

        reverify_sigopt_models_results = pd.concat([
            reverify_sigopt_models_results,
            new_row
        ], ignore_index=True)

    sigopt_name = build_sigopt_name("data/", target_prop, model_params["struct_type"], model_params["interpolation"], model_params["model_type"])
    exp_id = get_experiment_id(model_params, target_prop)
    save_directory = "./saved_models/" + model_params["model_type"] + "/"+ sigopt_name + "/" +str(exp_id)
    reverify_sigopt_models_results.to_csv(save_directory + "/reverify_sigopt_models_results.csv")


def keep_the_best_few_models(model_params, target_prop, num_best_models=3):
    sigopt_name = build_sigopt_name("data/", target_prop, model_params["struct_type"], model_params["interpolation"], model_params["model_type"])
    exp_id = get_experiment_id(model_params, target_prop)
    old_directory_prefix = "./saved_models/" + model_params["model_type"] + "/"+ sigopt_name + "/" +str(exp_id)
    new_directory_prefix = "./best_models/" + model_params["model_type"] + "/"+ sigopt_name + "/" +str(exp_id)

    reverify_sigopt_models_results = pd.read_csv(old_directory_prefix + '/reverify_sigopt_models_results.csv', index_col=0)

    for index, row in reverify_sigopt_models_results.iterrows():
        if not math.isclose(row['sigopt_loss'], row['reverified_loss'], rel_tol=1e-2):
            print("========\nSigopt model #" + str(index) + " has non-matching losses\n========")
    
    for i in range(num_best_models):
        new_directory = new_directory_prefix + "/best_" + str(i)
        if not os.path.exists(new_directory):
            os.makedirs(new_directory)

        folder_idx = reverify_sigopt_models_results.sort_values(by=['sigopt_loss']).index[i]
        old_directory = old_directory_prefix + "/observ_" + str(folder_idx)

        if model_params['model_type'] == "Painn":
            file_name = "best_model"
        else:
            file_name = "best_model.torch"

        shutil.copy(old_directory + "/" + file_name, new_directory + "/" + file_name)
        shutil.copy(old_directory + "/hyperparameters.json", new_directory + "/hyperparameters.json")
        print("Best models #" + str(i) + " found and saved")
