from models.Perovskite-Ordering-GCNNs-cgcnn.cgcnn.min import Normalizer
from models.Perovskite-Ordering-CGNNs-e3nn.utils.utils_model import Network
import torch_scatter

class PeriodicNetwork(Network):
    #### TAKEN FROM https://github.com/ninarina12/phononDoS_tutorial/blob/main/phononDoS.ipynb
    def __init__(self, in_dim, em_dim, **kwargs):            
        # override the `reduce_output` keyword to instead perform an averge over atom contributions    
        self.pool = False
        if kwargs['reduce_output'] == True:
            kwargs['reduce_output'] = False
            self.pool = True
            
        super().__init__(**kwargs)

        # embed the mass-weighted one-hot encoding
        self.em = nn.Linear(in_dim, em_dim)

    def forward(self, data: Union[tg.data.Data, Dict[str, torch.Tensor]]) -> torch.Tensor:
        data.x = F.relu(self.em(data.x))
        data.z = F.relu(self.em(data.z))
        output = super().forward(data)
        output = torch.relu(output)
        
        # if pool_nodes was set to True, use scatter_mean to aggregate
        if self.pool == True:
            output = torch_scatter.scatter_mean(output, data.batch, dim=0)  # take mean over atoms per example
        
        maxima, _ = torch.max(output, dim=1)
        output = output.div(maxima.unsqueeze(1))
        
        return output

def get_e3nn_model(hyperparameters, train_loader, is_contrastive = False):
    in_dim = 92
    em_dim = hyperparameters['len_embedding_feature_vector']
    out_dim = hyperparameters['num_hidden_feature']

    n_train_mean = get_neighbors(train_loader, is_contrastive)
    
    model = PeriodicNetwork(
        in_dim=in_dim,                                       # dimension of one-hot encoding of atom type
        em_dim=em_dim,                                       # dimension of atom-type embedding
        out_dim=out_dim,                                     # dimension of the embedding after covolution and pooling
        hid_dim = hyperparameters['num_hidden_feature'],         # number of features in each hidden layer
        n_hid = hyperparameters['num_hidden_layer'],             # number of hidden layers after covolution and pooling
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

    return model, Normalizer


def get_neighbors(train_loader, is_contrastive):
    # https://github.com/ninarina12/phononDoS_tutorial/blob/main/phononDoS.ipynb
    n = []

    if is_contrastive:
        for i, batch in enumerate(train_loader):
            for Compentry in batch:
                for data in CompEntry:
                    N = data.pos.shape[0]
                    for i in range(N):
                        n.append(len((data.edge_index[0] == i).nonzero()))

    else:

        for i, batch in enumerate(train_loader):
            for data in batch:
                N = data.pos.shape[0]
                for i in range(N):
                    n.append(len((data.edge_index[0] == i).nonzero()))

    return np.array(n).mean()