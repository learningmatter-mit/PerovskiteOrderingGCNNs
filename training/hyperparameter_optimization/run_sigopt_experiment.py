import sigopt
import sys
import pandas as pd
import argparse
import pickle as pkl
from processing.utils import filter_data_by_properties,select_structures
from processing.interpolation.Interpolation import *
from processing.dataloader.dataloader import get_dataloader
from training.hyperparameters.sigopt_parameters import *
from training.model_training.trainer import *
from training.evaluate import *

def run_sigopt_experiment(data_name,target_prop,is_relaxed,interpolation,model_type,gpu_num,experiment_id=None,sigopt_settings=None):

    training_data = pd.read_json(data_name + 'training_set.json')
    validation_data = pd.read_json(data_name + 'validation_set.json')
    test_data = pd.read_json(data_name + 'test_set.json')
    edge_data = pd.read_json(data_name + 'edge_dataset.json')

    if not interpolation:
        training_data = pd.concat((training_data,edge_data))

    data = [training_data, validation_data, test_data]

    for dataset in data:
        dataset = filter_data_by_properties(dataset,target_prop)

        if is_relaxed:
            dataset = select_structures(dataset,"relaxed")
        else:
            dataset = select_structures(dataset,"unrelaxed")

        if interpolation:
            dataset = apply_interpolation(dataset,target_prop)

    conn = sigopt.Connection(client_token="DIBWSINOFKSOTPFJQRSQCHRFWOYLYAJJDXEYFTBOBLPDLGVG")

    if experiment_id == None:
        experiment = create_sigopt_experiment(data_name,target_prop,is_relaxed,interpolation,model_type,sigopt_settings)
    else:
        experiment = conn.experiments(experiment_id).fetch()

    while experiment.progress.observation_count < experiment.observation_budget:
        print(experiment.progress.observation_count)
        
        suggestion = conn.experiments(experiment.id).suggestions().create()

        value = evaluate_metric(suggestion.assignments,data,target_prop,interpolation,model_type,experiment.progress.observation_count,gpu_num)    

        conn.experiments(experiment.id).observations().create(
            suggestion=suggestion.id,
            value=value,
        )
        
        experiment = conn.experiments(experiment.id).fetch()


def sigopt_evaluate_model(hyperparameters,data,target_prop,interpolation,model_type,observation_count,gpu_num):
    assert cuda == True
    device = "cuda:" + str(gpu_num)
    
    train_data = data[0]
    validation_data = data[1]
    test_data = data[2]

    train_loader = get_dataloader(train_data,target_prop,model_type,hyperparameters["batch_size"],interpolation)
    val_loader = get_dataloader(validation_data,target_prop,model_type,1,interpolation)
    test_loader = get_dataloader(test_data,target_prop,model_type,1,interpolation)
    
    model, normalizer = create_model(model_type,train_loader)

    model_save_dir = '/saved_models/'+ model_type + '/Scrap/observ_' + str(observation_count)
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir) 

    best_model,loss_fn = trainer(model,normalizer,model_type,train_loader,val_loader,hyperparameters,model_save_dir,gpu_num)
    
    best_loss = evaluate_model(model, normalizer, model_type, val_loader, loss_fn, gpu_num)

    return best_loss


def evaluate_metric(hyperparameters,data,target_prop,model_type,observation_count,gpu_num):
  best_error = sigopt_evaluate_model(hyperparameters,data, target_prop,model_type,observation_count,gpu_num)

  return best_error

def create_sigopt_experiment(data_name,target_prop,is_relaxed,model_type,sigopt_settings):
    sigopt_name = target_prop
    if is_relaxed:
        sigopt_name += "_" 
        sigopt_name += "relaxed"
    else:
        sigopt_name += "_" 
        sigopt_name += "unrelaxed"
    if interpolation:
        sigopt_name += "_" 
        sigopt_name += "interpolation"

    sigopt_name = sigopt_name + "_" + model_type

    if model_type == "Painn"
        curr_parameters = get_painn_hyperparameter_range()
    elif model_type == "CGCNN":
        curr_parameters = get_cgcnn_hyperparameter_range()
    else:
        curr_parameters = get_e3nn_hyperparameter_range()
    conn.experiments().create(
        name=sigopt_name, 
        parameters = curr_parameters,
        metrics=[dict(name="val_mae", objective="minimize", strategy="optimize")],
        observation_budget=sigopt_settings["obs_budget"], 
        parallel_bandwidth=sigopt_settings["parallel_band"],
    )
    return experiment


####### DEFINE MAIN

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Do Hyperparameter Optimization')
    parser.add_argument('--path', default = "data/", type=str, metavar='path',
                        help='the path to data')
    parser.add_argument('--prop', default = "dft_e_hull", type=str, metavar='name',
                        help='the property to predict')
    parser.add_argument('--relaxed', default = False, type=bool, metavar='representation',
                        help='the structure representation')
    parser.add_argument('--interpolation', default = True, type=bool, metavar='representation',
                        help='interpolation usage')
    parser.add_argument('--model', default = "e3nn", type="str", metavar='model',
                        help='the neural network to use')
    parser.add_argument('--gpu', default = 0, type=int, metavar='device',
                        help='the gpu to use')
    parser.add_argument('--id', default = -1, type=int, metavar='sigopt_props',
                        help='id for sigopt experiment')
    parser.add_argument('--parallel', default = 4, type=int, metavar='sigopt_props',
                        help='bandwidth of sigopt')
    parser.add_argument('--budget', default = 50, type=int, metavar='sigopt_props',
                        help='budget of sigopt')
    args = parser.parse_args()

    data_name = args.path
    target_prop = args.prop
    is_relaxed = args.relaxed
    interpolation = args.interpolation
    model_type = args.model
    gpu_num = args.gpu
    if args.id == -1:
        experiment_id = None
        sigopt_settings = {}
        sigopt_settings["parallel_band"] = args.parallel
        sigopt_settings["obs_budget"] = args.budget
    else:
        experiment_id = args.id
        sigopt_settings = None
    run_sigopt_experiment(data_name,target_prop,is_relaxed,interpolation,model_type,gpu_num,experiment_id,sigopt_settings)