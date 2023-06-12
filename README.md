# PerovskiteOrderingGCNNs

To clone this repo and all submodules:
```
git clone --recurse-submodules git@github.com:learningmatter-mit/PerovskiteOrderingGCNNs.git
```
or
```
git clone git@github.com:learningmatter-mit/PerovskiteOrderingGCNNs.git
git submodule update --init
```

To only update the submodules:
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
sigopt            8.8.0
gdown             4.7.1
mscorefonts       0.0.1
```

All these packages can be installed using the [environment.yml](environment.yml) file and `conda`:
```
conda env create -f environment.yml
conda activate Perovskite_ML_Environment
```
