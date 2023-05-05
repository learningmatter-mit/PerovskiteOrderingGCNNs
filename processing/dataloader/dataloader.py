import pandas as pd
from models.Perovskite-Ordering-CGCNNs-cgcnn.cgcnn.data import get_cgcnn_loader


def get_dataloaders(df, model_type, batch_size)
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
        train_loader, val_loader, test_loader = get_CGCNN_dataloaders(train_data,val_data,test_data,batch_size)
    elif model_type == "Painn":
        train_loader, val_loader, test_loader = get_Painn_dataloaders(train_data,val_data,test_data,batch_size)
    elif model_type == "e3nn":
        train_loader, val_loader, test_loader = get_e3nn_dataloaders(train_data,val_data,test_data,batch_size)
    elif model_type == "e3nn_contrastive":
        train_loader, val_loader, test_loader = get_e3nn_contrastive_dataloaders(train_data,val_data,test_data,batch_size)
    else:
        print("Model Type Not Supported")
        return None
    return train_loader, val_loader, test_loader

def get_CGCNN_dataloaders(train_data,val_data,test_data,batch_size):
    ### get_train_dataset
    train_loader = get_cgcnn_loader(train_data, batch_size=batch_size)
    ### get_val_dataset
    val_loader = get_cgcnn_loader(val_data, batch_size=1)   
    ### get_test_dataset
    test_loader = get_cgcnn_loader(test_data, batch_size=1)
    return train_loader, val_loader, test_loader

def get_Painn_dataloaders(train_data,val_data,test_data,batch_size):

def get_e3nn_dataloaders(train_data,val_data,test_data,batch_size):

def get_e3nn_contrastive_dataloaders(train_data,val_data,test_data,batch_size):
