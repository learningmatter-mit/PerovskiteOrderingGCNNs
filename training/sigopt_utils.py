def build_sigopt_name(target_prop,is_relaxed,interpolation,model_type):
    sigopt_name = target_prop

    if is_relaxed:
        sigopt_name += "_" 
        sigopt_name += "relaxed"
    else:
        sigopt_name += "_" 
        sigopt_name += "unrelaxed"

    if interpolation:
        sigopt_name += "_" 
        sigopt_name += "interpolation"

    sigopt_name = sigopt_name + "_" + model_type
    return sigopt_name