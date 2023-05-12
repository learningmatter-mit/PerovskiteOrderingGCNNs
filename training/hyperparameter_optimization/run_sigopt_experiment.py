import sigopt
import pickle as pkl
from processing.utils import filter_data_by_properties,select_structures
from processing.interpolation.Interpolation import *
from training.hyperparameters.sigopt_parameters import *
from training.model_training.trainer import *
from training.evaluate import *

def run_sigopt_experiment(data_name,target_prop,is_relaxed,model_type,gpu_num,experiment_id=None,sigopt_settings=None)
    data_path = "data/" + data_name

    file = open('data_path', 'rb')
    data = pkl.load(file)
    file.close()

    data = filter_data_by_properties(data,target_prop)
    if is_relaxed:
        data = select_structures(data,"relaxed")
    else:
        data = select_structures(data,"unrelaxed")
    data = apply_interpolation(data,"dft_e_hull")

    conn = sigopt.Connection(client_token="DIBWSINOFKSOTPFJQRSQCHRFWOYLYAJJDXEYFTBOBLPDLGVG")

    if experiment_id == None:
        experiment = create_sigopt_experiment(data_name,target_prop,is_relaxed,model_type,sigopt_settings)
    else:
        experiment = conn.experiments(experiment_id).fetch()

    while experiment.progress.observation_count < experiment.observation_budget:
        print(experiment.progress.observation_count)
        
        suggestion = conn.experiments(experiment.id).suggestions().create()

        value = evaluate_metric(suggestion.assignments,data,target_prop,model_type,experiment.progress.observation_count,gpu_num)    

        conn.experiments(experiment.id).observations().create(
            suggestion=suggestion.id,
            value=value,
        )
        
        experiment = conn.experiments(experiment.id).fetch()


def sigopt_evaluate_model(hyperparameters,data, target_prop,model_type,observation_count,gpu_num):
    assert cuda == True
    device = "cuda:" + str(gpu_num)
    
    train_loader, val_loader, test_loader = get_dataloaders(data,target_prop,model_type,hyperparameters["batch_size"])
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
    sigopt_name = data_name + "_" + target_prop + "_" is_relaxed + "_" + model_type
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
        observation_budget=sigopt_settings["obs_budge"], 
        parallel_bandwidth=sigopt_settings["parallel_band"],
    )
    return experiment


####### DEFINE MAIN