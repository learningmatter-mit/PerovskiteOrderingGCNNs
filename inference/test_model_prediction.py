import json
import torch
import pandas as pd
import numpy as np
from processing.dataloader.dataloader import get_dataloader
from processing.utils import filter_data_by_properties,select_structures
from processing.interpolation.Interpolation import *
from training.sigopt_utils import build_sigopt_name
from processing.create_model.create_model import create_model
from inference.select_best_models import get_experiment_id
from nff.train.loss import build_mae_loss
from nff.train.evaluate import evaluate
from torch.autograd import Variable


def get_all_model_predictions(model_params, gpu_num, target_prop, num_best_models=3):
    for test_set_type in ["test_set", "holdout_set_B_sites", "holdout_set_series"]:
        get_model_prediction(test_set_type, model_params, gpu_num, target_prop, num_best_models)
        print("Completed model prediction for " + test_set_type)


def get_model_prediction(test_set_type, model_params, gpu_num, target_prop, num_best_models):
    device_name = "cuda:" + str(gpu_num)
    device = torch.device(device_name)
    torch.cuda.set_device(device)

    interpolation = model_params["interpolation"]
    model_type = model_params["model_type"]    
    
    training_data = pd.read_json('data/' + 'training_set.json')
    test_data = pd.read_json('data/' + test_set_type + '.json')
    edge_data = pd.read_json('data/' + 'edge_dataset.json')    

    print("Loaded data")

    if not interpolation:
        training_data = pd.concat((training_data,edge_data))

    data = [training_data, test_data]
    processed_data = []

    for dataset in data:
        dataset = filter_data_by_properties(dataset,target_prop)
        dataset = select_structures(dataset,model_params["struct_type"])

        if interpolation:
            dataset = apply_interpolation(dataset,target_prop)

        processed_data.append(dataset)

    print("Completed data processing")

    train_data = processed_data[0]
    test_data = processed_data[1]

    train_loader = get_dataloader(train_data,target_prop,model_type,1,interpolation)
    test_loader = get_dataloader(test_data,target_prop,model_type,1,interpolation)       

    sigopt_name = build_sigopt_name("data/", target_prop, model_params["struct_type"], model_params["interpolation"], model_params["model_type"])
    exp_id = get_experiment_id(model_params, target_prop)

    for idx in range(num_best_models):
        directory = "./best_models/" + model_params["model_type"] + "/" + sigopt_name + "/" +str(exp_id) + "/" + "best_" + str(idx)
        model, normalizer = load_model(gpu_num, train_loader, model_params, directory)
        prediction = evaluate_model_with_tracked_ids(model, normalizer, gpu_num, test_loader, model_params)

        sorted_prediction = []
        infer_data = test_data.copy()
        infer_data.drop(columns=['structure', 'ase_structure'], inplace=True)
        if model_params["model_type"] == "e3nn":
            infer_data.drop(columns=['datapoint'], inplace=True)
            
        for index, _ in infer_data.iterrows():
            sorted_prediction.append(prediction[index])

        if interpolation:
            infer_data["predicted_"+target_prop+"_diff"] = sorted_prediction
            infer_data["predicted_" + target_prop] = infer_data["predicted_"+target_prop+"_diff"] + infer_data[target_prop + '_interp']
        else:
            infer_data["predicted_"+target_prop] = sorted_prediction

        infer_data.to_json(directory + '/' + test_set_type + "_predictions.json")

        
def load_model(gpu_num, train_loader, model_params, directory):
    device_name = "cuda:" + str(gpu_num)
    device = torch.device(device_name)

    with open(directory + "/hyperparameters.json") as json_file:
        assignments = json.load(json_file)

    if model_params["model_type"] == "Painn":
        model = torch.load(directory + "/best_model", map_location=device)
        normalizer = None
    else:
        model, normalizer = create_model(model_params["model_type"], train_loader, assignments)
        model.to(device)
        model.load_state_dict(torch.load(directory + "/best_model.torch", map_location=device)['state'])
    
    return model, normalizer


def evaluate_model_with_tracked_ids(model, normalizer, gpu_num, test_loader, model_params):
    device_name = "cuda:" + str(gpu_num)
    device = torch.device(device_name)
    predictions = {}

    if model_params["model_type"] == "Painn":
        prop_names = model.output_keys
        loss_fn_painn = build_mae_loss(loss_coef = {target_prop: 1.0 for target_prop in prop_names})
        results, targets, _ = evaluate(model, 
                                          test_loader, 
                                          loss_fn_painn, 
                                          device=gpu_num)
        
        prop_name = prop_names[0]
        out = [float(entry) for entry in results[prop_name]]
        ids = list([int(i) for i in targets['crystal_id']])
        for i in range(len(ids)):
            predictions[ids[i]] = out[i]

        return predictions

    else:
        model.eval()    
        with torch.no_grad():
            for j, d in enumerate(test_loader):
                if model_params["model_type"] == "CGCNN":
                    input_struct = d[0]
                    target = d[1]
                    input_var = (Variable(input_struct[0].cuda(non_blocking=True)),
                                 Variable(input_struct[1].cuda(non_blocking=True)),
                                 input_struct[2].cuda(non_blocking=True),
                                 [crys_idx.cuda(non_blocking=True) for crys_idx in input_struct[3]])
                    output = model(*input_var).view(-1)
                    target = Variable(target.cuda(non_blocking=True))
                    crys_idx = d[2]
                else:
                    d.to(device)
                    output = model(d)
                    crys_idx = d.idx

                predictions_iter = normalizer.denorm(output).detach().cpu().numpy().reshape(-1)
                try:
                    crys_idx = crys_idx.detach().cpu().numpy().reshape(-1)
                except:
                    crys_idx = np.array(crys_idx)

                for i in range(crys_idx.shape[0]):
                    predictions[int(crys_idx[i])] = predictions_iter[i]

        return predictions
