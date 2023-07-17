import sigopt
import sys
import pandas as pd
import argparse
import pickle as pkl
import torch
import numpy as np
import random
import shutil
from step_2_run_sigopt_experiment import run_sigopt_experiment

task_idx_to_settings = {

    1: {
        "data_name": "pretrain_data/",
        "target_prop": "decomposition_energy",
        "model": "CGCNN",
        "gpu": 0,
        "nickname": "test_1",
        "structure_type": "unrelaxed",
        "interpolation": True,
        "id": -1,
    },
    2: {
        "data_name": "pretrain_data/",
        "target_prop": "decomposition_energy",
        "model": "CGCNN",
        "gpu": 0,
        "nickname": "test_2",
        "structure_type": "spud",
        "interpolation": True,
        "id": -1,
    },

    3: {
        "data_name": "pretrain_data/",
        "target_prop": "decomposition_energy",
        "model": "e3nn",
        "gpu": 1,
        "nickname": "test_3",
        "structure_type": "unrelaxed",
        "interpolation": True,
        "id": -1,
    },

    4: {
        "data_name": "pretrain_data/",
        "target_prop": "decomposition_energy",
        "model": "e3nn",
        "gpu": 1,
        "nickname": "test_4",
        "structure_type": "spud",
        "interpolation": True,
        "id": -1,
    },

}


if __name__ == '__main__':


    my_task_id = int(sys.argv[1])
    num_tasks = int(sys.argv[2])
    settings = task_idx_to_settings[my_task_id]

    data_name = settings["data_name"]
    target_prop = settings["target_prop"]
    model_type = settings["model"]
    gpu_num = settings["gpu"]
    nickname = settings["nickname"]
    struct_type = settings["structure_type"]
    interpolation = settings["interpolation"]
    
    if struct_type not in ["unrelaxed","relaxed","spud","M3Gnet_relaxed"]:
        raise ValueError('struct type is not available')  
        
    if settings["id"] == -1:
        experiment_id = None
        sigopt_settings = {}
        sigopt_settings["parallel_band"] = 4
        sigopt_settings["obs_budget"] = 50
    else:
        experiment_id = settings["id"]
        sigopt_settings = None

    run_sigopt_experiment(data_name,target_prop,struct_type,interpolation,model_type,gpu_num,experiment_id,sigopt_settings,nickname)