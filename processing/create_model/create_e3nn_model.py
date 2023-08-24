from typing import Union, Dict
from models.PerovskiteOrderingGCNNs_cgcnn.cgcnn.model import Normalizer
from models.PerovskiteOrderingGCNNs_e3nn.utils.utils_model import Network
from training.hyperparameters.default import get_default_e3nn_hyperparameters
import torch_scatter
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch_geometric as tg

class PeriodicNetwork(Network):
    def __init__(self, in_dim, em_dim, out_dim, hid_dim, n_hid, per_site = per_site, **kwargs):            
        # override the `reduce_output` keyword to instead perform an averge over atom contributions    
        self.pool = False
        if kwargs['reduce_output'] == True:
            kwargs['reduce_output'] = False
            self.pool = True

        self.per_site = per_site
            
        super().__init__(**kwargs)

        # embed the mass-weighted one-hot encoding
        self.em = nn.Linear(in_dim, em_dim)

        if n_hid > 0:
            self.conv_to_fc = torch.nn.Linear(out_dim, hid_dim)
            self.conv_to_fc_relu = torch.nn.ReLU()
            
            if n_hid > 1:
                self.fcs = torch.nn.ModuleList([torch.nn.Linear(hid_dim, hid_dim) for _ in range(n_hid-1)])
                self.relus = torch.nn.ModuleList([torch.nn.ReLU() for _ in range(n_hid-1)])
        
            self.fc_out = torch.nn.Linear(hid_dim, 1)
        
        else:
            self.fc_out = torch.nn.Linear(out_dim, 1)

    def forward(self, data: Union[tg.data.Data, Dict[str, torch.Tensor]]) -> torch.Tensor:
        data.x = F.relu(self.em(data.x))
        data.z = F.relu(self.em(data.z))
        atom_fea = torch.nn.functional.relu(super().forward(data))
        #assert self.pool == True
        if self.per_site:
            crys_fea = atom_fea
        else:
            crys_fea = torch_scatter.scatter_mean(atom_fea, data.batch, dim=0)       
               
        if hasattr(self, 'conv_to_fc') and hasattr(self, 'conv_to_fc_relu'):
            crys_fea = self.conv_to_fc_relu(self.conv_to_fc(crys_fea))
            
            if hasattr(self, 'fcs') and hasattr(self, 'relus'):
                for fc, relu in zip(self.fcs, self.relus):
                    crys_fea = relu(fc(crys_fea))        
        
        output =  self.fc_out(crys_fea)

        if self.per_site:
            output = output.view(data.num_graphs,-1)
        
        return output

def get_e3nn_model(hyperparameters, train_loader, per_site = False):

    if hyperparameters == "default":
        hyperparameters = get_default_e3nn_hyperparameters()

    in_dim = 92
    r_max = 5.0
    em_dim = hyperparameters['len_embedding_feature_vector']
    out_dim = hyperparameters['num_hidden_feature']

    n_train_mean = get_neighbors(train_loader)
    
    model = PeriodicNetwork(
        in_dim=in_dim,                                       # dimension of one-hot encoding of atom type
        em_dim=em_dim,                                       # dimension of atom-type embedding
        out_dim=out_dim,                                     # dimension of the embedding after covolution and pooling
        hid_dim = hyperparameters['num_hidden_feature'],         # number of features in each hidden layer
        n_hid = hyperparameters['num_hidden_layer'],             # number of hidden layers after covolution and pooling
        per_site = per_site,
        irreps_in=str(em_dim)+"x0e",                         # em_dim scalars (L=0 and even parity) on each atom to represent atom type
        irreps_out=str(out_dim)+"x0e",                       # out_dim scalars (L=0 and even parity) to output
        irreps_node_attr=str(em_dim)+"x0e",                  # em_dim scalars (L=0 and even parity) on each atom to represent atom type
        mul=hyperparameters['multiplicity_irreps'],          
        layers=hyperparameters['num_conv'] - 1,                  # number of nonlinearities (number of convolutions = layers + 1)
        number_of_basis=hyperparameters['num_radical_basis'],    # number of basis on which the edge length are projected
        radial_neurons=hyperparameters['num_radial_neurons'],    # number of neurons in the hidden layers of the radial fully connected network
        lmax=2,                                              # maximum order of spherical harmonics
        max_radius=r_max,                                    # cutoff radius for convolution
        num_neighbors=n_train_mean,                        # scaling factor based on the typical number of neighbors
        reduce_output=True                                   # whether or not to aggregate features of all atoms at the end
    )

    sample_target = []
    for i, temp in enumerate(train_loader):
        sample_target.append(temp.target.view(-1))
    sample_target = np.concatenate(sample_target).ravel()
    normalizer = Normalizer(torch.tensor(sample_target))

    model.pool = True
    return model, normalizer


def get_neighbors(train_loader):
    # https://github.com/ninarina12/phononDoS_tutorial/blob/main/phononDoS.ipynb
    n = []

    for i, batch in enumerate(train_loader):
        N = batch.pos.shape[0]
        for i in range(N):
            n.append(len((batch.edge_index[0] == i).nonzero()))

    return np.array(n).mean()