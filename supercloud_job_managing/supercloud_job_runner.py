
import sys
sys.path.insert(0, '/home/gridsan/jdamewood/perovskites/PerovskiteOrderingGCNNs')
import sigopt
import os
import pandas as pd
import argparse
import pickle as pkl
import torch
import numpy as np
import random
import json
import shutil
from processing.utils import filter_data_by_properties,select_structures
from processing.interpolation.Interpolation import *
from processing.dataloader.dataloader import get_dataloader
from processing.create_model.create_model import create_model
from training.hyperparameters.sigopt_parameters import *
from training.model_training.trainer import *
from training.sigopt_utils import build_sigopt_name
from training.evaluate import *


def supercloud_run_job(data_name,hyperparameters,target_prop,struct_type,interpolation,model_type,gpu_num,experiment_id,nickname):

    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    if data_name == "data/":

        training_data = pd.read_json(data_name + 'training_set.json')
        validation_data = pd.read_json(data_name + 'validation_set.json')
        edge_data = pd.read_json(data_name + 'edge_dataset.json')

        if not interpolation:
            training_data = pd.concat((training_data,edge_data))

    elif data_name == "pretrain_data/":

        training_data = pd.read_json(data_name + 'training_set.json')
        validation_data = pd.read_json(data_name + 'validation_set.json')

    else:
        print("Specified Data Directory Does Not Exist!")

    print("Loaded data")

    data = [training_data, validation_data]
    processed_data = []

    for dataset in data:
        dataset = filter_data_by_properties(dataset,target_prop)

        dataset = select_structures(dataset,struct_type)

        if interpolation:
            dataset = apply_interpolation(dataset,target_prop)

        processed_data.append(dataset)

    print("Completed data processing")

    supercloud_evaluate_model(data_name,hyperparameters,processed_data,target_prop,interpolation,model_type,experiment_id,gpu_num,nickname)


def supercloud_evaluate_model(data_name,hyperparameters,processed_data,target_prop,interpolation,model_type,experiment_id,gpu_num,nickname):
    device_name = "cuda:" + str(gpu_num)
    device = torch.device(device_name)
    torch.cuda.set_device(device)

    train_data = processed_data[0]
    validation_data = processed_data[1]

    train_loader = get_dataloader(train_data,target_prop,model_type,hyperparameters["batch_size"],interpolation)
    train_eval_loader = None

    if "e3nn" in model_type and "pretrain" not in data_name:
        train_eval_loader = get_dataloader(train_data,target_prop,"e3nn_contrastive",1,interpolation)
        val_loader = get_dataloader(validation_data,target_prop,"e3nn_contrastive",1,interpolation)
    else:
        val_loader = get_dataloader(validation_data,target_prop,model_type,1,interpolation)
    
    model, normalizer = create_model(model_type,train_loader,hyperparameters)
    
    sigopt_name = build_sigopt_name(data_name,target_prop,struct_type,interpolation,model_type)
    model_tmp_dir = './saved_models/'+ model_type + '/' + sigopt_name + '/' + str(experiment_id) + '/' + nickname + '_tmp' + str(gpu_num)
    if os.path.exists(model_tmp_dir):
        shutil.rmtree(model_tmp_dir)
    os.makedirs(model_tmp_dir) 

    best_model,loss_fn = trainer(model,normalizer,model_type,train_loader,val_loader,hyperparameters,model_tmp_dir,gpu_num,train_eval_loader)
    
    is_contrastive = False
    if "contrastive" in model_type:
        is_contrastive = True
    _, _, best_loss = evaluate_model(best_model, normalizer, model_type, val_loader, loss_fn, gpu_num,is_contrastive=is_contrastive)

    if model_type != "Painn":
        best_loss = best_loss[0]
 
    training_results = {"validation_loss": best_loss}

    with open(model_tmp_dir + "/training_results.json", "w") as outfile:
        json.dump(training_results, outfile)

    



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NN training parameters')
    parser.add_argument('--experiment_group_name', type=str,help="the name of the experiment group", required=True,)
    parser.add_argument('--experiment_sigopt_id', type=str,help="the id of the particular experiment", required=True,)
    parser.add_argument('--suggestion_num',type=str,help="the suggestion number", required=True,)

    args = parser.parse_args()
    
    gpu_num = str(0)

    experiment_group_name = args.experiment_group_name

    experiment_id = args.experiment_sigopt_id

    suggestion_num = args.suggestion_num

    f = open("supercloud_job_managing/experiments/" +experiment_group_name+ "/settings.json")
    experimental_group_settings = json.load(f)
    f.close()

    f = open("supercloud_job_managing/experiments/" +experiment_group_name+ "/sigopt_info.json")
    sigopt_info = json.load(f)
    f.close()

    data_name = sigopt_info[experiment_id]["settings"]["data_name"]
    target_prop = sigopt_info[experiment_id]["settings"]["target_prop"]
    struct_type = sigopt_info[experiment_id]["settings"]["struct_type"]
    interpolation = sigopt_info[experiment_id]["settings"]["interpolation"]
    model_type = sigopt_info[experiment_id]["settings"]["model_type"]

    hyperparameters = sigopt_info[experiment_id]["observations"]["temporary"][suggestion_num]["hyperparameters"]

    nickname = str(suggestion_num)

    f = open("supercloud_job_managing/experiments/" +experiment_group_name+ "/sigopt_info.json","w")
    json.dump(sigopt_info, f)
    f.close()

    supercloud_run_job(data_name,hyperparameters,target_prop,struct_type,interpolation,model_type,gpu_num,experiment_id,nickname)
    
    f = open("supercloud_job_managing/experiments/" +experiment_group_name+ "/sigopt_info.json")
    sigopt_info = json.load(f)
    f.close()

    sigopt_info[experiment_id]["observations"]["temporary"][suggestion_num]["status"] = "completed"

    f = open("supercloud_job_managing/experiments/" +experiment_group_name+ "/sigopt_info.json","w")
    json.dump(sigopt_info, f)
    f.close()

