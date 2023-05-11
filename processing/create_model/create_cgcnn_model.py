from models.PerovskiteOrderingGCNNs_cgcnn.cgcnn.model import CrystalGraphConvNet, Normalizer

def get_cgcnn_model(hyperparameters,train_loader):

    training_labels = []
    
    for i, (struct, target, _) in enumerate(train_loader):
        
        training_labels.append(target.view(-1,1))
        
    training_labels = np.concatenate(training_labels).ravel()
    normalizer = Normalizer(torch.tensor(training_labels))
    
    orig_atom_fea_len = struct[0].shape[-1]
    nbr_fea_len = struct[1].shape[-1]
    CGC_model = CrystalGraphConvNet(orig_atom_fea_len, nbr_fea_len,
                                atom_fea_len=hyperparameters["atom_fea_len"],
                                n_conv=hyperparameters["n_conv"],
                                h_fea_len=hyperparameters["h_fea_len"],
                                n_h=hyperparameters["n_h"],
                                classification=False)

    return CGC_model, normalizer