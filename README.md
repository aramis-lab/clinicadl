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
git clone git@github.com:aramis-lab/AD-DL.git
cd clinicadl
pip install -e .
```

## Use in command line mode

```
clinicadl -h

usage: clinicadl [-h] {preprocessing,extract,train,classify} ...

Clinica Deep Learning.

optional arguments: -h, --help            show this help message and exit

Task to execute with clinicadl: 
  What kind of task do you want to use with clinicadl (preprocessing, 
  extract, train, validate, classify).

  {preprocessing,extract,train,classify} 
                        Stages/task to execute with clinicadl
    preprocessing       Prepare data for training (needs clinica instqlled).
    extract             Create data (slices or patches) for training.
    train               Train with your data and create a model.
    classify            Classify one image or a list of images with your
                        previouly trained model.  

```

Typical use for preprocessing:

```bash
clinicadl preprocessing --np 32 \
  $BIDS_DIR \
  $CAPS_DIR \
  $TSV_FILE \
  $REF_TEMPLATE \
  $WORKING_DIR
```


## Or use the scripts
```
python run_train.py --max_steps 10000 --dropout_rate 0.2
```
# run testing:
```
python run_test.py
```
