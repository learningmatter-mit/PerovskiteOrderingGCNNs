from procssing.create_model.create_cgcnn_model import get_cgcnn_model
from procssing.create_model.create_cgcnn_model import get_painn_model
from procssing.create_model.create_cgcnn_model import get_e3nn_model


def create_model(model_name, hyperparameters, train_loader):

    if model_name == "CGCNN":
        return get_cgcnn_model(hyperparameters, train_loader)
    elif model_name == "Painn":
        return get_painn_model(hyperparameters, train_loader)
    elif model_name == "e3nn":
        return get_e3nn_model(hyperparameters, train_loader)
    elif model_name == "e3nn_contrastive":
        return get_e3nn_model(hyperparameters, train_loader)
    else:
        print("Model Type Not Supported")

