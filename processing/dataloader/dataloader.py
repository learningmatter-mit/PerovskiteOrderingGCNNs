import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler

from models.Perovskite-Ordering-CGCNNs-cgcnn.cgcnn.data import get_cgcnn_loader
from models.Perovskite-Ordering-CGCNNs-painn.nff.data import Dataset, collate_dicts

def get_dataloaders(df, prop, model_type, batch_size)
    pd.options.mode.chained_assignment = None # Disable the SettingWithCopy warning (due to pandas.apply as new column)
    df['ase_structure'] = df.progress_apply(lambda x: AseAtomsAdaptor.get_atoms(x['structure']), axis=1)
    df['idx'] = df_interp.index

    df_all_multi = copy.deepcopy(df)

    split_seed = 0
    train_ids = df_all_multi.sample(frac=0.6,random_state=split_seed).index.tolist()
    val_ids = df_all_multi[~df_all_multi.index.isin(train_ids)].sample(frac=0.5,random_state=split_seed).index.tolist()
    test_ids = df_all_multi[(~df_all_multi.index.isin(train_ids))&(~df_all_multi.index.isin(val_ids))].index.tolist()

    train_data = df[df.index.isin(train_ids)].copy()
    val_data = df[df.index.isin(val_ids)].copy()
    test_data = df[df.index.isin(test_ids)].copy()

    if model_type == "CGCNN":
        train_loader, val_loader, test_loader = get_CGCNN_dataloaders(train_data,val_data,test_data,prop,batch_size)
    elif model_type == "Painn":
        train_loader, val_loader, test_loader = get_Painn_dataloaders(train_data,val_data,test_data,prop,batch_size)
    elif model_type == "e3nn":
        train_loader, val_loader, test_loader = get_e3nn_dataloaders(train_data,val_data,test_data,prop,batch_size)
    elif model_type == "e3nn_contrastive":
        train_loader, val_loader, test_loader = get_e3nn_contrastive_dataloaders(train_data,val_data,test_data,prop,batch_size)
    else:
        print("Model Type Not Supported")
        return None
    return train_loader, val_loader, test_loader

def get_CGCNN_dataloaders(train_data,val_data,test_data,prop,batch_size):
    ### get_train_dataset
    train_loader = get_cgcnn_loader(train_data, prop, batch_size=batch_size)
    ### get_val_dataset
    val_loader = get_cgcnn_loader(val_data, prop, batch_size=1)   
    ### get_test_dataset
    test_loader = get_cgcnn_loader(test_data, prop, batch_size=1)
    return train_loader, val_loader, test_loader

def get_Painn_dataloaders(train_data,val_data,test_data,prop,batch_size):

    train_props = dataframe_to_props_painn(train_data, prop)
    val_props = dataframe_to_props_painn(val_data, prop)
    test_props = dataframe_to_props_painn(test_data, prop)

    train_dataset = Dataset(train_props, units='eV', stack=True)
    val_dataset = Dataset(val_props, units='eV', stack=True)
    test_dataset = Dataset(test_props, units='eV', stack=True)

    f = open("atom_init.json")
    atom_inits = json.load(f)

    for key, value in atom_inits.items():
        atom_inits[key] = np.array(value, dtype=np.float32)

    for dataset in [train_dataset, val_dataset, test_dataset]:
        dataset.generate_neighbor_list(cutoff=5.0, undirected=False)
        dataset.generate_atom_initializations(atom_inits)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_dicts, 
                          sampler=RandomSampler(train_dataset))
    val_loader = DataLoader(val_dataset, batch_size=1, collate_fn=collate_dicts)
    test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_dicts)

    return train_loader, val_loader, test_loader

def get_e3nn_dataloaders(train_data,val_data,test_data,batch_size):

def get_e3nn_contrastive_dataloaders(train_data,val_data,test_data,batch_size):


def dataframe_to_props_painn(df, target_prop):   
    prop_names = [target_prop+"_diff"]
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