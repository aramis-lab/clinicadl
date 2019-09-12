# Clinica Deep Learning AD
This repository contains a software framework for reproducible experiments with
convolutional neural networks on automatic classification of Alzheimer's
disease (AD) using anatomical MRI data from the publicly available dataset
ADNI. It is developed by Junhao WEN, Elina Thibeau--Sutre and Mauricio DIAZ MELO.
The preprint of the corresponding paper may be found [here](https://arxiv.org/abs/1904.07773)


# Bibliography
All the papers described in the State of the art section of the manuscript may be found at this URL address: <https://www.zotero.org/groups/2337160/ad-dl>.


# Dependencies:
- Clinica
- Pytorch
- Nilearn
- Nipy

# How to use ?

## Create a conda environment with the corresponding dependencies:

```
conda create --name clincadl_env_py36 python=3.6 jupyter
conda activate clinicadl_env_py36
conda install -c aramislab -c conda-forge clinica
pip install -r requirements.txt
conda install -c pytorch pytorch torchvision
```

## Install the package `clinicadl` as developer:

```
pip install -e .Code/clinicadl/
```

## Use in command line mode

```
clinicadl preprocessing \
  -bd=/bids/directory \
  -cd=/caps/directory \
  -tsv=file.tsv \
  -rs=template.nii \
  -wd=/working/dir
```

Tape `clinicadl --help` for more info.


## Or use the scripts
```
python run_train.py --max_steps 10000 --dropout_rate 0.2
```
# run testing:
```
python run_test.py
```
