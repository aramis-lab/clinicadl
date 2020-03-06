# Clinica Deep Learning AD
This repository contains a software framework for reproducible experiments with
convolutional neural networks (CNN) on automatic classification of Alzheimer's
disease (AD) using anatomical MRI data from the publicly available dataset
ADNI. It is developed by Junhao Wen, Elina Thibeau--Sutre and Mauricio Diaz.
The preprint version of the corresponding paper may be found
[here](https://arxiv.org/abs/1904.07773).

Automatic Classification of AD using a classical machine learning approach can
be performed using the software available here:
<https://github.com/aramis-lab/AD-ML>.

This software is currently in *active developmet*.
Pretrained models for the CNN networks can be obtained here:
<https://zenodo.org/record/3491003>  

# Bibliography
All the papers described in the State of the art section of the manuscript may
be found at this URL address: <https://www.zotero.org/groups/2337160/ad-dl>.


# Dependencies:
- Python >= 3.6
- Clinica (needs only to perform preprocessing) >= 0.3.2
- Numpy
- Pandas
- Scikit-learn
- Pandas
- Pytorch => 1.1
- Nilearn == 0.5.2
- Nipy
- TensorBoardX

# How to use?

## Create a conda environment with the corresponding dependencies:

```
conda create --name clinicadl_env python=3.6 jupyter
conda activate clinicadl_env
conda install -c aramislab -c conda-forge clinica

git clone git@github.com:aramis-lab/AD-DL.git
cd AD-DL
pip install -r requirements.txt
conda install -c pytorch pytorch torchvision
```

## Install the package `clinicadl` as developer:

```
cd clinicadl
pip install -e .
```

## Use in command line mode

```bash
clinicadl -h

usage: clinicadl [-h] {preprocessing,extract,train,classify} ...

Clinica Deep Learning.

optional arguments: -h, --help            show this help message and exit

Task to execute with clinicadl:
  What kind of task do you want to use with clinicadl (preprocessing,
  extract, train, validate, classify).

  {preprocessing,extract,train,classify}
                        Stages/task to execute with clinicadl
    preprocessing       Prepare data for training (needs clinica installed).
    extract             Create data (slices or patches) for training.
    train               Train with your data and create a model.
    classify            Classify one image or a list of images with your
                        previouly trained model.  
```

Typical use for `preprocessing`:

```bash
clinicadl preprocessing --np 32 \
  $BIDS_DIR \
  $CAPS_DIR \
  $TSV_FILE \
  $WORKING_DIR
```

For detailed instructions type `clinica 'action' -h`.
For example:

```bash
clinicadl train -h
usage: clinicadl train [-h] [-gpu] [-np NPROC] [--batch_size BATCH_SIZE]
                       [--evaluation_steps EVALUATION_STEPS]
                       [--preprocessing {linear,mni}]
                       [--diagnoses DIAGNOSES [DIAGNOSES ...]] [--baseline]
                       [--minmaxnormalization] [--n_splits N_SPLITS]
                       [--split SPLIT] [-tAE]
                       [--accumulation_steps ACCUMULATION_STEPS]
                       [--epochs EPOCHS] [--learning_rate LEARNING_RATE]
                       [--patience PATIENCE] [--tolerance TOLERANCE]
                       [--add_sigmoid] [--pretrained_path PRETRAINED_PATH]
                       [--pretrained_difference PRETRAINED_DIFFERENCE]
                       {subject,slice,patch,svm} caps_directory tsv_path
                       output_dir network

positional arguments:
  {subject,slice,patch,svm}
                        Choose your mode (subject level, slice level, patch
                        level, svm).
  caps_directory        Data using CAPS structure.
  tsv_path              tsv path with sujets/sessions to process.
  output_dir            Folder containing results of the training.
  network               CNN Model to be used during the training.

optional arguments:
  -h, --help            show this help message and exit
  -gpu, --use_gpu       Uses gpu instead of cpu if cuda is available
  -np NPROC, --nproc NPROC
                        Number of cores used during the training
  --batch_size BATCH_SIZE
                        Batch size for training. (default=2)
  --evaluation_steps EVALUATION_STEPS, -esteps EVALUATION_STEPS
                        Fix the number of batches to use before validation
  --preprocessing {linear,mni}
                        Defines the type of preprocessing of CAPS data.
  --diagnoses DIAGNOSES [DIAGNOSES ...], -d DIAGNOSES [DIAGNOSES ...]
                        Take all the subjects possible for autoencoder
                        training
  --baseline            if True only the baseline is used
  --minmaxnormalization, -n
                        Performs MinMaxNormalization
  --n_splits N_SPLITS   If a value is given will load data of a k-fold CV
  --split SPLIT         Will load the specific split wanted.
  -tAE, --train_autoencoder
                        Add this option if you want to train an autoencoder
  --accumulation_steps ACCUMULATION_STEPS, -asteps ACCUMULATION_STEPS
                        Accumulates gradients in order to increase the size of
                        the batch
  --epochs EPOCHS       Epochs through the data. (default=20)
  --learning_rate LEARNING_RATE, -lr LEARNING_RATE
                        Learning rate of the optimization. (default=0.01)
  --patience PATIENCE   Waiting time for early stopping.
  --tolerance TOLERANCE
                        Tolerance value for the early stopping.
  --add_sigmoid         Ad sigmoid function at the end of the decoder.
```

## Or use the scripts
Look at the `clinicadl/scripts/` folder.

# Run testing:
```
pytest clinicadl/tests/
```
