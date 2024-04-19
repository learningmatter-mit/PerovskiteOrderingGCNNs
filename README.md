# PerovskiteOrderingGCNNs

Repo for our paper "Learning Orderings in Crystalline Materials with Symmetry-Aware Graph Neural Networks".

## Setup

To start, clone this repo and all its submodules to your local directory or a workstation:

```
git clone --recurse-submodules git@github.com:learningmatter-mit/PerovskiteOrderingGCNNs.git
```

or

```
git clone git@github.com:learningmatter-mit/PerovskiteOrderingGCNNs.git
cd PerovskiteOrderingGCNNs
git submodule update --init
```

Our codes are built upon previous implementations of [CGCNN](https://github.com/learningmatter-mit/PerovskiteOrderingGCNNs_cgcnn/tree/af4c0bf6606da1b46887ed8c29521d199d5e2798), [e3nn](https://github.com/learningmatter-mit/PerovskiteOrderingGCNNs_e3nn/tree/408b90e922a2a9c7bae2ad95433aae97d1a58494), and [PaiNN](https://github.com/learningmatter-mit/PerovskiteOrderingGCNNs_painn/tree/e7980a52af4936addc5fb03dbc50d4fc74fe98fc), which are included as submodules in this repo. If there are any changes in their corresponding GitHub repos, the following command will update the submodules in this repo:

```
git submodule update --remote --merge
```

This repository requires the following packages to run correctly:

```
pandas            1.5.3
scipy             1.10.1
numpy             1.24.3
scikit-learn      1.2.2
matplotlib        3.7.1
seaborn           0.12.2
pymatgen          2023.5.10
ase               3.22.1
rdkit             2023.3.1
e3fp              1.2.5
pytorch           1.13.1
pytorch-cuda      11.7
pytorch-sparse    0.6.17
pytorch-scatter   2.1.1
pytorch-cluster   1.6.1
torchvision       0.14.1
torchaudio        0.13.1
pyg               2.3.0
e3nn              0.5.1
sigopt            8.8.2
sigoptlite        0.1.2
gdown             4.7.1
mscorefonts       0.0.1
boken             3.3.4
```

All these packages can be installed using the [environment.yml](environment.yml) file and `conda`:

```
conda env create -f environment.yml
conda activate Perovskite_ML_Environment
```

## Citation
If you use our codes, data, and/or models, please cite the following paper:
```
TBD
```
