def build_sigopt_name(data_source,target_prop,struct_type,interpolation,model_type):
    sigopt_name = target_prop

    if data_source == "data/":
        sigopt_name += "_" 
        sigopt_name += "htvs_data"

    elif data_source == "pretrain_data/":
        sigopt_name += "_" 
        sigopt_name += "pretrain_data"

    sigopt_name += "_" 
    sigopt_name += struct_type

    if interpolation:
        sigopt_name += "_" 
        sigopt_name += "interpolation"

    sigopt_name = sigopt_name + "_" + model_type
    return sigopt_name