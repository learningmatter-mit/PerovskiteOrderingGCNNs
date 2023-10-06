
import argparse
from processing.utils import filter_data_by_properties,select_structures
from processing.interpolation.Interpolation import *
from processing.dataloader.dataloader import get_dataloader
from inference.test_model_prediction import *
from training.model_training.trainer import *

def run_retrain_sigopt_experiment(model_path,retrain_name,target_prop,struct_type,interpolation,model_type,contrastive_weight,gpu_num,experiment_id,sigopt_settings):

    processed_data = get_processed_data(target_prop,interpolation)
    print("Completed data processing")

    conn = sigopt.Connection(client_token="ERZVVPFNRCSCQIJVOVCYHVFUGZCKUDPOWZJTEXZYNOMRKQLS")
    name = "Refining_" + retrain_name

    if experiment_id == None:
        experiment = create_retrain_sigopt_experiment(retrain_name,sigopt_settings,conn)
        print("Created a new retraining SigOpt experiment with ID: " + str(experiment.id))
    else:
        experiment = conn.experiments(experiment_id).fetch()
        print("Continuing a prior retraining SigOpt experiment with ID: " + str(experiment.id))

    while experiment.progress.observation_count < experiment.observation_budget:
        print('\n========================\nSigopt experiment count #', experiment.progress.observation_count)
        
        suggestion = conn.experiments(experiment.id).suggestions().create()

        value = sigopt_evaluate_retrain_model(model_path,retrain_name,suggestion.assignments,processed_data,target_prop,interpolation,model_type,contrastive_weight,experiment.id,experiment.progress.observation_count,gpu_num)    

        conn.experiments(experiment.id).observations().create(
            suggestion=suggestion.id,
            value=value,
        )

        experiment = conn.experiments(experiment.id).fetch()
        observation_id = experiment.progress.observation_count - 1

        model_save_dir = './saved_models/Refining/' + retrain_name + '/' + str(experiment.id) + '/observ_' + str(observation_id)
        model_tmp_dir = './saved_models/Refining/' + retrain_name + '/' + str(experiment.id) + '/tmp_' + str(gpu_num)

        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)

        ### Copy contents of tmp file
        possible_file_names = ["best_model", "best_model.pth.tar", "best_model.torch",
                               "final_model.torch","final_model","final_model.pth.tar",
                               "log_human_read.csv","checkpoints/checkpoint-100.pth.tar"]
        for file_name in possible_file_names:
            if os.path.isfile(model_tmp_dir + "/" + file_name):
                if file_name == "checkpoints/checkpoint-100.pth.tar":
                    shutil.move(model_tmp_dir + "/" + file_name, model_save_dir + "/" + "checkpoint-100.pth.tar")
                else:
                    shutil.move(model_tmp_dir + "/" + file_name, model_save_dir + "/" + file_name)
        
        ### Empty tmp file
        shutil.rmtree(model_tmp_dir)

        torch.cuda.empty_cache()

def get_processed_data(prop,interpolation):
    training_data = pd.read_json('data/training_set.json')
    validation_data = pd.read_json('data/validation_set.json')
    edge_data = pd.read_json('data/edge_dataset.json')

    if not interpolation:
        training_data = pd.concat((training_data,edge_data))

    data = [training_data, validation_data]
    processed_data = []

    for dataset in data:
        dataset = filter_data_by_properties(dataset,target_prop)

        dataset = select_structures(dataset,struct_type)

        if interpolation:
            dataset = apply_interpolation(dataset,target_prop)

        processed_data.append(dataset)

    return processed_data


def sigopt_evaluate_retrain_model(model_path,retrain_name,hyperparameters,processed_data,target_prop,interpolation,model_type,contrastive_weight,experiment.id,experiment.progress.observation_count,gpu_num):
    device = "cuda:" + str(gpu_num)
    
    train_data = processed_data[0]
    validation_data = processed_data[1]

    train_loader = get_dataloader(train_data,target_prop,model_type,hyperparameters["batch_size"],interpolation)
    train_eval_loader = None

    if "e3nn" in model_type:
        train_eval_loader = get_dataloader(train_data,target_prop,"e3nn_contrastive",1,interpolation)
        val_loader = get_dataloader(validation_data,target_prop,"e3nn_contrastive",1,interpolation)
    else:
        val_loader = get_dataloader(validation_data,target_prop,model_type,1,interpolation)
    
    model_params = {}
    model_params["interpolation"] = interpolation
    model_params["model_type"] = model_type
    model, normalizer = load_model(gpu_num, train_loader, model_params, model_path, target_prop)
    
    model_tmp_dir = './saved_models/Refining/' + retrain_name + '/' + str(experiment.id) + '/tmp_' + str(gpu_num)

    if os.path.exists(model_tmp_dir):
        shutil.rmtree(model_tmp_dir)
    os.makedirs(model_tmp_dir) 

    best_model,loss_fn = trainer(model,normalizer,model_type,train_loader,val_loader,hyperparameters,model_tmp_dir,gpu_num,train_eval_loader=train_eval_loader,contrastive_weight=contrastive_weight)
    
    is_contrastive = False
    if "contrastive" in model_type:
        is_contrastive = True
    _, _, best_loss = evaluate_model(best_model, normalizer, model_type, val_loader, loss_fn, gpu_num,is_contrastive=is_contrastive, contrastive_weight=contrastive_weight)

    if model_type == "Painn":
        return best_loss
    else:
        return best_loss[0]


def create_sigopt_experiment(retrain_name,sigopt_settings,conn):


    curr_parameters = get_retrain_hyperparameter_range()

    experiment = conn.experiments().create(
        name=retrain_name, 
        parameters = curr_parameters,
        metrics=[dict(name="val_mae", objective="minimize", strategy="optimize")],
        observation_budget=sigopt_settings["obs_budget"], 
        parallel_bandwidth=sigopt_settings["parallel_band"],
    )
    return experiment


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Refining Models for perovksite ordering GCNNs')
    parser.add_argument('--model_path', required=True, type=str, metavar='device',
                        help="path to the model to retrain")
    parser.add_argument('--retrain_name', required=True, type=str, metavar='device',
                        help="sigopt name of the model to retrain")
    parser.add_argument('--prop', default = "dft_e_hull", type=str, metavar='name',
                        help="the property to predict (default: dft_e_hull; other options: Op_band_center)")
    parser.add_argument('--struct_type', default = 'unrelaxed', type=str, metavar='struct_type',
                        help="using which structure representation (default: unrelaxed; other options: relaxed, M3Gnet_relaxed)")
    parser.add_argument('--interpolation', default = 'no', type=str, metavar='yes/no',
                        help="using interpolation (default: no; other options: yes)")
    parser.add_argument('--model', default = "CGCNN", type=str, metavar='model',
                        help="the neural network to use (default: CGCNN; other options: Painn, e3nn, e3nn_contrastive)")
    parser.add_argument('--contrastive_weight', default = 1.0, type=float, metavar='loss_parameters',
                        help="the weighting applied to the contrastive loss term (default: 1.0)")
    parser.add_argument('--gpu', default = 0, type=int, metavar='device',
                        help="the gpu to use (default: 0)")
    parser.add_argument('--id', default = -1, type=int, metavar='sigopt_props',
                        help="id for sigopt experiment (default: -1)")
    parser.add_argument('--parallel', default = 4, type=int, metavar='sigopt_props',
                        help="bandwidth of sigopt (default: 4)")
    parser.add_argument('--budget', default = 20, type=int, metavar='sigopt_props',
                        help="budget of sigopt (default: 20)")


    args = parser.parse_args()

    path = args.model_path 
    retrain_name = args.retrain_name
    target_prop = args.prop
    model_type = args.model
    gpu_num = args.gpu
    struct_type = args.struct_type
    contrastive_weight = args.contrastive_weight
    if args.interpolation == 'yes':
        interpolation = True
    elif args.interpolation == 'no':
        interpolation = False
    else:
        raise ValueError('interpolation needs to be yes or no') 


    if args.id == -1:
        experiment_id = None
        sigopt_settings = {}
        sigopt_settings["parallel_band"] = args.parallel
        sigopt_settings["obs_budget"] = args.budget
    else:
        experiment_id = args.id
        sigopt_settings = None

    run_retrain_sigopt_experiment(model_path,retrain_name,target_prop,struct_type,interpolation,model_type,contrastive_weight,gpu_num,experiment_id,sigopt_settings)
    