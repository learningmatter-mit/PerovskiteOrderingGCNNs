def get_default_cgcnn_hyperparameters():

    hyperparameters = {}

    hyperparameters['MaxEpochs'] = 100
    hyperparameters["log_lr"] = -3
    hyperparameters["reduceLR_patience"] = 10
    hyperparameters["atom_fea_len"] = 64
    hyperparameters["n_conv"] = 3
    hyperparameters["h_fea_len"] = 128
    hyperparameters["n_h"] = 1

    return hyperparameters

def get_default_painn_hyperparameters():

    hyperparameters = {}

    hyperparameters['MaxEpochs'] = 100
    hyperparameters["log_lr"] = -3
    hyperparameters["reduceLR_patience"] = 10
    hyperparameters["log2_feat_dim"] = 6
    hyperparameters["activation"] = "ReLU"
    hyperparameters["num_conv"] = 3

    return hyperparameters


def get_default_e3nn_hyperparameters():

    hyperparameters = {}

    hyperparameters['MaxEpochs'] = 100
    hyperparameters["log_lr"] = -3
    hyperparameters["reduceLR_patience"] = 10
    hyperparameters['len_embedding_feature_vector'] = 64
    hyperparameters['num_hidden_feature'] = 128   
    hyperparameters['num_hidden_layer'] = 2
    hyperparameters['multiplicity_irreps'] = 32      
    hyperparameters['num_conv'] = 3      
    hyperparameters['num_radical_basis'] = 10
    hyperparameters['num_radial_neurons'] = 100

    return hyperparameters