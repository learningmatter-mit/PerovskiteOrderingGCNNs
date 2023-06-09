import torch
import json
import pandas as pd
import numpy as np
from processing.utils import filter_data_by_properties,select_structures
from processing.interpolation.Interpolation import apply_interpolation
from training.sigopt_utils import build_sigopt_name
from processing.dataloader.dataloader import get_dataloader
from processing.create_model.create_model import create_model
from training.loss import contrastive_loss
from training.evaluate import evaluate_model


def predict(data_name,model_params,prop,gpu_num=0,num_models=3):

    data_src = "data/" + data_name + ".json"
    infer_data = pd.read_json(data_src)
    train_data = pd.read_json('data/training_set.json')

    model_type = model_params["model_type"]
    interpolation = model_params["interpolation"]
    is_relaxed = model_params["relaxed"]

    data = [train_data, infer_data]
    processed_data = []
    for dataset in data:
        dataset = filter_data_by_properties(dataset,prop)

        if is_relaxed:
            dataset = select_structures(dataset,"relaxed")
        else:
            dataset = select_structures(dataset,"unrelaxed")

        if interpolation:
            dataset = apply_interpolation(dataset,prop)

        processed_data.append(dataset)

    train_data = processed_data[0]
    infer_data = processed_data[1]

    train_loader = get_dataloader(train_data, prop, model_type, 1, interpolation)
    infer_loader = get_dataloader(infer_data, prop, model_type, 1, interpolation)

    predictions = get_predictions(train_loader, infer_loader, model_params, prop, gpu_num, num_models)

    mean_predictions = []

    for index, row in infer_data.iterrows():
        mean_predictions.append(np.mean(predictions[index]))
    
    if interpolation:
        infer_data["predicted_diff_"+prop] = mean_predictions
        infer_data["predicted_" + prop] = infer_data["predicted_diff_"+prop] + infer_data[prop + '_interp']
    else:
        infer_data["predicted_"+prop] = mean_predictions

    return infer_data

def get_predictions(train_loader, infer_loader, model_params, prop, gpu_num, num_models):
    device_name = "cuda:" + str(gpu_num)
    device = torch.device(device_name)
    torch.cuda.set_device(device)
    predictions = {}

    for i in range(num_models):
        ### get model
        model, normalizer = load_model(train_loader, model_params, prop, i)
        model = model.to(device)

        curr_predictions = evaluate_model_ids(model,normalizer, model_params["model_type"],infer_loader,gpu_num)

        for idx in curr_predictions:
            if idx in predictions:
                predictions[idx].append(curr_prediction[idx])
            else:
                predictions[idx] = [curr_prediction[idx]]

    for idx in predictions:
        predictions[idx] = np.asarray(predictions[idx])

    return predictions


def load_model(train_loader, model_settings, prop, i):
    sigopt_name = build_sigopt_name(prop,model_settings["relaxed"],model_settings["interpolation"],model_settings["model_type"])
    directory = "./saved_models/best_models/" + sigopt_name + "/" + "best_model_" + str(i)

    ### get assignments
    f = open(directory + '/assignments.json')
    assignments = json.load(f)
    f.close()

    if model_type == "Painn":
        model = torch.load(directory + "/best_model", map_location=device)
        normalizer = None
    else:
        model, normalizer = create_model(model_type,train_loader,assignments)
        model.load_state_dict(torch.load(directory + "/best_model.torch", map_location=device)['state'])
    

    return model, normalizer



def evaluate_model_ids(model, normalizer, model_type, dataloader, gpu_num):
    device_name = "cuda:" + str(gpu_num)
    device = torch.device(device_name)

    if model_type == "Painn":
        prop_names = model.output_keys
        loss_fn_painn = build_mae_loss(loss_coef = {prop: 1.0 for prop in prop_names})
        results, targets, loss = evaluate(model, 
                                          data_loader, 
                                          loss_fn_painn, 
                                          device=gpu_num)
        predictions = {}
        prop_name = prop_names[0]
        out = [float(entry) for entry in results[prop_name]]
        targ = [float(entry) for entry in targets[prop_name]]
        ids = list([int(i) for i in targets['crystal_id']])

        for i in range(len(ids)):
            predictions[ids[i]] = out[i]

        return predictions

    model.eval()
    predictions = {}
   
    with torch.no_grad():
        for j, d in enumerate(dataloader):
            if model_type == "CGCNN":
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
            crys_idx.detach().cpu().numpy().reshape(-1)
            
            for i in range(crys_idx.shape[0]):
                predictions[int(crys_idx[i])] = predictions_iter[i]
          
    return predictions
