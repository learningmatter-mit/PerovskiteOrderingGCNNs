from processing.create_model.create_cgcnn_model import get_cgcnn_model
from processing.create_model.create_painn_model import get_painn_model
from processing.create_model.create_e3nn_model import get_e3nn_model


def create_model(model_name, train_loader, interpolation, prop, hyperparameters="default",per_site=False):

    if model_name == "CGCNN":
        return get_cgcnn_model(hyperparameters,train_loader,per_site=per_site)
    elif model_name == "Painn":
        return get_painn_model(hyperparameters, train_loader, interpolation, prop)
    elif (model_name == "e3nn") or (model_name == "e3nn_contrastive"):
        return get_e3nn_model(hyperparameters, train_loader,per_site=per_site)
    else:
        print("Model Type Not Supported")

