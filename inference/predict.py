import json
import numpy as np
import sigopt
from processing.utils import filter_data_by_properties,select_structures


def predict(data_name,model_params,prop):

    data_src = "data/" + data_name + ".json"
    data = pd.read_json(data_src)

    model_type = model_params["model_type"]
    interpolation = model_params["interpolation"]
    is_relaxed = model_params["relaxed"]

    data = filter_data_by_properties(data,prop)

    if interpolation:
        data = apply_interpolation(data,prop)



    return None

