def build_sigopt_name(data_source,target_prop,struct_type,interpolation,model_type,contrastive_weight=1.0,training_fraction=1.0,training_seed=0,long_range=False):
    sigopt_name = target_prop

    if data_source == "data/":
        sigopt_name += "_" 
        sigopt_name += "htvs_data"

    elif data_source == "pretrain_data/":
        sigopt_name += "_" 
        sigopt_name += "pretrain_data"
        
    elif data_source == "data_per_site/":
        sigopt_name += "_" 
        sigopt_name += "data_per_site"

    elif data_source == "data_per_site/":
        sigopt_name += "_" 
        sigopt_name += "data_per_site"

    sigopt_name += "_" 
    sigopt_name += struct_type

    if interpolation:
        sigopt_name += "_" 
        sigopt_name += "interpolation"

    sigopt_name = sigopt_name + "_" + model_type
    
    if long_range:
        sigopt_name += "_"
        sigopt_name += "Long_Range"

    if long_range:
        sigopt_name += "_" 
        sigopt_name += "Long_Range"

    if contrastive_weight != 1.0:
        sigopt_name += "_" 
        sigopt_name += "ContrastiveWeight"
        sigopt_name += str(contrastive_weight)

    if training_fraction != 1.0:
        sigopt_name += "_" 
        sigopt_name += "TrainingFraction"
        sigopt_name += str(training_fraction)

    if training_seed != 0:
        sigopt_name += "_" 
        sigopt_name += "TrainingSeed"
        sigopt_name += str(training_seed)

    return sigopt_name