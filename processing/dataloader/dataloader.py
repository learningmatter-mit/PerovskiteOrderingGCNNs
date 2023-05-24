import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler
import torch_geometric as tg
import json
from tqdm import tqdm
from pymatgen.io.ase import AseAtomsAdaptor
from processing.dataloader.build_data import build_e3nn_data, construct_contrastive_dataset
from processing.dataloader.contrastive_data import CompDataLoader
from models.PerovskiteOrderingGCNNs_cgcnn.cgcnn.data import get_cgcnn_loader
import sys
sys.path.append('models/PerovskiteOrderingGCNNs_painn/')
from nff.data import Dataset, collate_dicts


def get_dataloader(data, prop="dft_e_hull", model_type="cgcnn", batch_size=10, interpolation=True):
    tqdm.pandas()
    pd.options.mode.chained_assignment = None # Disable the SettingWithCopy warning (due to pandas.apply as new column)
    
    data['ase_structure'] = data.progress_apply(lambda x: AseAtomsAdaptor.get_atoms(x['structure']), axis=1)
    data['idx'] = data.index

    if interpolation:
        prop += "_diff"

    if model_type == "cgcnn":
        data_loader = get_cgcnn_loader(data,prop,batch_size)
    elif model_type == "painn":
        data_loader = get_painn_dataloader(data,prop,batch_size)
    elif model_type == "e3nn":
        data_loader = get_e3nn_dataloader(data,prop,batch_size)
    elif model_type == "e3nn_contrastive":
        data_loader = get_e3nn_contrastive_dataloader(data,prop,batch_size)
    else:
        raise ValueError("Model Type Not Supported")

    return data_loader


def get_painn_dataloader(data,prop,batch_size):

    data_props = dataframe_to_props_painn(data, prop)
    dataset = Dataset(data_props, units='eV', stack=True)
    f = open("processing/dataloader/atom_init.json")
    atom_inits = json.load(f)

    for key, value in atom_inits.items():
        atom_inits[key] = np.array(value, dtype=np.float32)

    dataset.generate_neighbor_list(cutoff=5.0, undirected=False)
    dataset.generate_atom_initializations(atom_inits)
    data_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_dicts, sampler=RandomSampler(dataset))

    return data_loader


def get_e3nn_dataloader(data,prop,batch_size):
    data['datapoint'] = data.progress_apply(lambda x: build_e3nn_data(x, prop, r_max=5.0), axis=1)
    data_loader = tg.loader.DataLoader(data['datapoint'].values, batch_size=batch_size, shuffle=True)

    return data_loader


def get_e3nn_contrastive_dataloader(data,prop,batch_size):
    comp_data = construct_contrastive_dataset(data,prop,r_max=5.0)
    data_loader = CompDataLoader(comp_data, batch_size=batch_size, shuffle=True)

    return data_loader


def dataframe_to_props_painn(df, target_prop):   
    prop_names = [target_prop]
    props = {}
    id_list = []
    nxyz_list = []
    props_list = {prop: [] for prop in prop_names}
    lattice_list = []    
    
    for index, row in df.iterrows():
        curr_struct = row['ase_structure']
        id_list.append(index)
        
        for prop in prop_names:
            props_list[prop].append(row[prop])
        
        n = np.asarray(curr_struct.numbers).reshape(-1,1)
        xyz = np.asarray(curr_struct.positions)
        curr_nxyz = np.concatenate((n, xyz), axis=1)
        nxyz_list.append(curr_nxyz)
        lattice_list.append(curr_struct.cell[:])
        
    props['crystal_id'] = id_list
    props['nxyz'] = nxyz_list
    
    for prop in prop_names:
        props[prop] = props_list[prop]
    
    props['lattice'] = lattice_list
        
    return props
