def get_cgcnn_hyperparameter_range():
    parameters=[

        # dict(name='MaxEpochs', bounds=dict(min=100, max=100), type="int"), 
        dict(name="batch_size", bounds=dict(min=4, max=16), type="int"),
        dict(name="log_lr", bounds=dict(min=-5, max=-2), type="int"), 
        dict(name="reduceLR_patience", bounds=dict(min=10, max=30), type="int"),

        dict(name="atom_fea_len", bounds=dict(min=32, max=256), type="int"), 
        dict(name="n_conv", bounds=dict(min=2, max=5), type="int"), 
        dict(name="h_fea_len", bounds=dict(min=32, max=256), type="int"), 
        dict(name="n_h", bounds=dict(min=1, max=4), type="int")
    ]
    return parameters


def get_painn_hyperparameter_range():
    parameters=[
        # dict(name='MaxEpochs', bounds=dict(min=100, max=100), type="int"), 
        dict(name="batch_size", bounds=dict(min=4, max=16), type="int"),
        dict(name="log_lr", bounds=dict(min=-5, max=-2), type="int"), 
        dict(name="reduceLR_patience", bounds=dict(min=10, max=30), type="int"), 

        dict(name="log2_feat_dim", bounds=dict(min=5, max=9), type="int"), 
        dict(name="activation", categorical_values=["swish", "learnable_swish", "ReLU", "LeakyReLU"], type="categorical"),
        dict(name="num_conv", bounds=dict(min=1, max=6), type="int"), 
     ]
    return parameters


def get_e3nn_hyperparameter_range():
    parameters=[

        # dict(name='MaxEpochs', bounds=dict(min=100, max=100), type="int"), 
        dict(name="batch_size", bounds=dict(min=4, max=12), type="int"),
        dict(name="log_lr", bounds=dict(min=-5, max=-2), type="int"), 
        dict(name="reduceLR_patience", bounds=dict(min=10, max=30), type="int"), 

        dict(name="len_embedding_feature_vector", grid=[32, 64, 128], type="int"),
        dict(name="num_hidden_feature", grid=[32, 64, 128], type="int"), 
        dict(name="num_hidden_layer", bounds=dict(min=0, max=2), type="int"), 
        dict(name="multiplicity_irreps", grid=[16, 32, 64], type="int"),
        dict(name="num_conv", bounds=dict(min=1, max=4), type="int"), 
        dict(name="num_radical_basis", grid=[5, 10, 20], type="int"),
        dict(name="num_radial_neurons", grid=[32, 64, 128], type="int"), 
    ]
    return parameters