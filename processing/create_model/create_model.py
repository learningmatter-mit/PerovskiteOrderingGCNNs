from processing.create_model.create_cgcnn_model import get_cgcnn_model
from processing.create_model.create_painn_model import get_painn_model
from processing.create_model.create_e3nn_model import get_e3nn_model


def create_model(model_name, train_loader, hyperparameters="default", prop="dft_e_hull"):

    if model_name == "CGCNN":
        return get_cgcnn_model(hyperparameters,train_loader)
    elif model_name == "Painn":
        return get_painn_model(hyperparameters, train_loader,prop)
    elif (model_name == "e3nn") or (model_name == "e3nn_contrastive"):
        return get_e3nn_model(hyperparameters, train_loader)
    else:
        print("Model Type Not Supported")

