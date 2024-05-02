# PerovskiteOrderingGCNNs
[![arXiv](https://img.shields.io/badge/arXiv-TBD-red.svg)]()
[![Zenodo](https://img.shields.io/badge/Zenodo-10.5281/zenodo.11099803-blue.svg)](https://doi.org/10.5281/zenodo.11099803)
[![MDF](https://img.shields.io/badge/Materials_Data_Facility-10.18126/3hyv--5a38-purple.svg)]([https://doi.org/10.18126/3hyv-5a38)
[![MIT](https://img.shields.io/badge/License-MIT-black.svg)](https://opensource.org/license/mit)

Repo for our paper "Learning Orderings in Crystalline Materials with Symmetry-Aware Graph Neural Networks". This repo is currently under construction.

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

## Usage

All our data and pretrained models are archived on Zenodo ([DOI: 10.5281/zenodo.11099803](https://doi.org/10.5281/zenodo.11099803)) and Materials Data Facility ([DOI: 10.18126/3hyv-5a38](https://doi.org/10.18126/3hyv-5a38)). Please place all data and model files in the corresponding directories and then refer to the following Jupyter notebooks to reproduce the results of our paper:

- [1_model_training.ipynb](1_model_training.ipynb): This notebook provides examples of how to train GCNNs on the training dataset and conduct hyperparameter optimization based on the loss on the validation set.
- [2_model_inference.ipynb](2_model_inference.ipynb): This notebook provides examples of how to verify the performance of GCNNs on the validation set, select the top-performing models accordingly, compute the prediction on the test and holdout sets, and extract the latent embeddings of CGCNN and e3nn after all message passing and graph convolution layers.
- [3_model_analysis.ipynb](3_model_analysis.ipynb): This notebook provides examples of how to reproduce all major figures in this manuscript.

## Citation
If you use our codes, data, and/or models, please cite the following paper:
```
TBD
```
