# Quickstart

## What is ClinicaDL?

Motivations:
- Accessible entry point for deep learning in neuorimaging (high-level, less complex than Pytorch)
- But still flexible (Python API): enabled experienced users to build complex workflows
- Focus on experiment management and reproducibility -> useful for benchmarking
- Well integrated in medical imaging (MONAI, TorchIO) and neuroimaging community (BIDS, Clinica (https://aramislab.paris.inria.fr/clinica/docs/public/latest/))
## Prerequisite 

BIDS (data must be organized in BIDS), PyTorch
and have a look at MONAI and TorchIO
## 10 Minutes to ClinicaDL

A summary of the user guide that follows.

# 1. Manipulating neuorimaging data

## 1.1 Data structures

``clinicadl.data.structures.DataPoint`` (necessity to keep an image, associated mask and metadata in the same object)
``clinicadl.data.structures.Sample`` and ``clinicadl.data.structures.Sample2D`` (child of DataPoint, output of a dataset, make a transition with the next part)

## 1.2 Reading BIDS datasets

``clinicadl.io.bids.Bids``
``clinicadl.io.bids.BidsFileType``
``clinicadl.data.datasets.BidsDataset``
### 1.2.1 Converting NIfTI images to tensors

To speed up dataloading.
``clinicadl.data.datasets.BidsDataset.to_tensors`` and ``clinicadl.data.datasets.TensorDataset``.
### 1.2.2 Joining multiple BIDS datasets

``clinicadl.data.datasets.ConcatDataset``, ``clinicadl.data.datasets.PairedDataset``, ``clinicadl.data.datasets.UnpairedDataset``
## 1.3 Transforming data

``clinicadl.transforms.TransformsHandler``
### 1.3.1 Patches and slices

``clinicadl.transforms.extraction``
### 1.3.2 Preprocessing, data augmentation, and post-processing

ClinicaDL works with any functions that takes as input and return a ``clinicadl.data.structures.DataPoint`` .
## 1.4 Splitting data

Importance of longitudinal split in neuorimaging to avoid data leakage.
### 1.4.1 Making a split

``clinicadl.split.make_split``, ``clinicadl.split.make_kfold``
### 1.4.2 Reading a split

``clinicadl.split.SingleSplit``, ``clinicadl.split.KFold``, ``clinicadl.split.Split``

## 1.5 Batching data for training

``clinicadl.data.dataloader`` (``DataLoader``, ``Batch``,  ``CollateFn``)

# 2. Building a deep learning workflow

## 2.1 Defining a model

``clinicadl.models`` (losses and optimizers comes from PyTorch; mention also ``clinicadl.infer`` for the evaluation)
### 2.1.1 Neural networks

``clinicadl.networks.nn``: many neural networks available
## 2.2 Training

``clinicadl.train`` (do not mention ``clinicadl.train.Trainer.validate`` and ``clinicadl.train.Trainer.test``)
Mention briefly the metrics but refer to ##2.3
### 2.2.1 Resuming an interrupted training

``clinicadl.train.Trainer.resume``
## 2.3 Evaluating

``clinicadl.metrics.Metric``, and ``clinicadl.metrics.MetricsHandler`` (briefly)
Also a quick reference to ``clinicadl.infer`` for customizing the inference/evaluation steps
``clinicadl.train.Trainer.validate`` and ``clinicadl.train.Trainer.test``
## 2.4 Callbacks

``clinicadl.callbacks``

# 3. Experiment management and reproducibility

## 3.1 Configuration classes

Config class: a dataclass associated to an object that will only contain the parameters of this object (almost no Python logic inside)
Easy to save and read -> good to reproduce experiment

Can access the underlying object with ``get_object`` method

To ensure the best reproducibility it is advised to work with config classes instead than raw objects.

The list of available config classes is not exhaustive (mostly associated to object from MONAI and TorchIO), but will be enriched continuously

## 3.2 MAPS

A MAPS contains all the outputs of training and evaluation phases --> useful for experiment management (all the results in the same folder, easy to share)
But it is also useful for reproducibility: contains all the hyperparameters used (warning: if object from outside of ClinicaDL (e.g., TorchIO, MONAI) it will work
but not be 100% reproducible)

# 4. Customising your ClinicaDL experiment

Mention callbacks briefly (make a reference to ##2.4)

Object-oriented programming: for many objects in ClinicaDL, it is possible inherit from a class to modify its behaviour (e.g., Model)