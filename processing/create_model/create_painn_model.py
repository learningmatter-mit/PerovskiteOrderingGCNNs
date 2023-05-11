from models.PerovskiteOrderingGCNNs_painn.nff.train import get_model
from training.hyperparameters.default import get_default_painn_hyperparameters
import numpy as np

def get_painn_model(hyperparameters,train_loader,prop):

    if hyperparameters = "default":
        hyperparameters = get_default_painn_hyperparameters()

    prop_name = prop+ "_diff"

    training_labels = []
    
    for i, data in enumerate(train_loader):
        
        training_labels.append(data[prop_name])

    training_data = np.concatenate(training_labels).ravel()

    modelparams = {"feat_dim": 2**hyperparameters["log2_feat_dim"], 
                   "activation": hyperparameters["activation"], 
                   "n_rbf": 20,
                   "cutoff": 5.0,
                   "num_conv": hyperparameters["num_conv"], 
                   "output_keys": [prop_name], 
                   "grad_keys": [], 
                   "skip_connection": {prop_name: False}, 
                   "learnable_k": False, 
                   "conv_dropout": 0.0, 
                   "readout_dropout": 0.0, 
                   "means": {prop_name: np.nanmean(training_data).item()}, 
                   "stddevs": {prop_name: np.nanstd(training_data).item()}
                  }

        
    model = get_model(modelparams, model_type="Painn")

    return model, None