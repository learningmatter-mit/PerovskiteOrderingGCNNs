import torch_geometric as tg
import torch
import json
import numpy as np
import random
from tqdm import tqdm
from ase import Atom
from ase.neighborlist import neighbor_list

from processing.dataloader.contrastive_data import CompData

default_dtype = torch.float32


def get_atom_encoding():
    # one-hot encoding atom type and mass (from Mingda Li, et al. Adv. Sci. 2021, 8, 2004214)
    type_encoding = {}
    specie_am = []
    for Z in range(1, 119):
        specie = Atom(Z)
        type_encoding[specie.symbol] = Z - 1
        specie_am.append(specie.mass)

    atom_inits_atomic_mass = torch.diag(torch.tensor(specie_am))

    # CGCNN-type embedding
    f = open("processing/dataloader/atom_init.json")
    atom_inits = json.load(f)

    atom_inits_cgcnn = []
    for i in range(1, 101):
        atom_inits_cgcnn.append(atom_inits[str(i)])
    atom_inits_cgcnn = torch.tensor(atom_inits_cgcnn, dtype=default_dtype)

    return type_encoding, atom_inits_cgcnn, atom_inits_atomic_mass


def build_e3nn_data(entry, prop, r_max, per_site = False):
    #### TAKEN FROM https://github.com/ninarina12/phononDoS_tutorial/blob/main/phononDoS.ipynb
    torch.set_default_dtype(default_dtype)
    type_encoding, atom_inits_cgcnn, atom_inits_atomic_mass = get_atom_encoding()
    symbols = list(entry['ase_structure'].symbols).copy()
    positions = torch.from_numpy(entry['ase_structure'].positions.copy()).float()
    lattice = torch.from_numpy(entry['ase_structure'].cell.array.copy()).float().unsqueeze(0)
        
    # edge_src and edge_dst are the indices of the central and neighboring atom, respectively
    # edge_shift indicates whether the neighbors are in different images or copies of the unit cell
    edge_src, edge_dst, edge_shift = neighbor_list("ijS", a=entry['ase_structure'], cutoff=r_max, self_interaction=True)
    #print(edge_src)
    #print(edge_dst)
    #print(edge_shift)

    # compute the relative distances and unit cell shifts from periodic boundaries
    edge_batch = positions.new_zeros(positions.shape[0], dtype=torch.long)[torch.from_numpy(edge_src)]
    #print(edge_batch)
    #print(positions[torch.from_numpy(edge_dst)].type())
    #print(torch.tensor(edge_shift).type())
    #print(lattice[edge_batch].type())
    edge_vec = (positions[torch.from_numpy(edge_dst)]
                - positions[torch.from_numpy(edge_src)]
                + torch.einsum('ni,nij->nj', torch.tensor(edge_shift, dtype = default_dtype), lattice[edge_batch]))

    #print(edge_vec)
    #print(edge_vec.type())
    
    # compute edge lengths (rounded only for plotting purposes)
    edge_len = np.around(edge_vec.norm(dim=1).numpy(), decimals=2)
        
    if "formula" in entry:
        curr_comp = entry["formula"]
    else:
        curr_comp = None

    if per_site:
        target_data = torch.tensor(entry[prop]).unsqueeze(0)
    else:
        target_data = torch.tensor([entry[prop]]).unsqueeze(0)

    data = tg.data.Data(
        pos=positions, lattice=lattice, symbol=symbols,
        comp = curr_comp,
        x=atom_inits_cgcnn[[type_encoding[specie] for specie in symbols]], # CGCNN-type embedding (node feature)
        z=atom_inits_cgcnn[[type_encoding[specie] for specie in symbols]], # CGCNN-type embedding (node attribute)
        edge_index=torch.stack([torch.LongTensor(edge_src), torch.LongTensor(edge_dst)], dim=0),
        edge_shift=torch.tensor(edge_shift, dtype=default_dtype),
        edge_vec=edge_vec,
        edge_len=edge_len,
        target=target_data,
        idx=torch.tensor([entry['idx']]).unsqueeze(0)
    )
        
    return data


def construct_contrastive_dataset(df,prop,r_max):   
    comp_to_data = {}
    for _,row in tqdm(df.iterrows(), total=df.shape[0]):     
        curr_data = build_e3nn_data(row, prop, r_max)
        
        if row.formula in comp_to_data:
            comp_to_data[row.formula].append(curr_data)
        else:
            comp_to_data[row.formula] = [curr_data]
                
    stored_data = []
    for formula in comp_to_data.keys():
        random.Random(0).shuffle(comp_to_data[formula])
        curr_arr = []
        for datapoint in comp_to_data[formula]:           
            curr_arr.append(datapoint)
            
            if len(curr_arr)==6:
                stored_data.append(CompData(curr_arr))
                curr_arr = []
        
        if len(curr_arr)>0:
            stored_data.append(CompData(curr_arr))
            
    return stored_data
