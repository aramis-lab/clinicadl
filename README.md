# Clinica Deep Learning AD
This repository contains a software framework for reproducible experiments with
2D convolutional neural network on automatic classification of Alzheimer's
disease (AD) using anatomical MRI data from the publicly available datasets
ADNI. It is developed by Junhao WEN and Ouyang WEI.  This architecture relies
heavily on the Clinica software platform that you will need to install. Another
prerequisite is to do image processing for the original MRI data by using
Clinica, then we fit the processed data into this CNN.

# Projects
- 1) Implement 2D CNN to fit the processed png images from MRI
- 2) Compress the 3D MRI to 2D multi-chanel image (compressive sensing), then
  fit into the 2D CNN.

# Dependencies:
- Clinica
- Pytorch
- Nilearn
- Nipy

# How to use ?

## Create a conda environment with the corresponding dependencies:

```
conda create --name clincadl_env_py27 python=2.7 jupyter
conda activate clinicadl_env_py27
conda install -c aramislab -c conda-forge clinica=0.1.3
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
