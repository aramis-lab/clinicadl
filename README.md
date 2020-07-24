<h1 align="center">
  Clinica Deep Learning (<i>clinicadl</i>)
</h1>

<p align="center"><strong>for Alzheimer's Disease</strong></p>

<p align="center">
  <a href="https://ci.inria.fr/clinicadl/job/AD-DL/job/master/">
    <img src="https://ci.inria.fr/clinicadl/buildStatus/icon?job=AD-DL%2Fmaster" alt="Build Status">
  </a>
  <a href="https://badge.fury.io/py/clinicadl">
    <img src="https://badge.fury.io/py/clinicadl.svg" alt="PyPI version">
  </a>
  <a href='https://clinicadl.readthedocs.io/en/latest/?badge=latest'>
    <img src='https://readthedocs.org/projects/clinicadl/badge/?version=latest' alt='Documentation Status' />
  </a>
</p>

<p align="center">
  See also:
  <a href="#related-repositories">AD-ML</a>,
  <a href="#related-repositories">Clinica</a>
</p>


## About the project

This repository hosts the code source for reproducible experiments on
**automatic classification of Alzheimer's disease (AD) using anatomical MRI
data**.
It allows to train convolutional neural networks (CNN) models.
The journal version of the paper describing this work is available
[here](https://doi.org/10.1016/j.media.2020.101694).

Automatic classification of AD using a classical machine learning approach can
be performed using the software available here:
<https://github.com/aramis-lab/AD-ML>.

> **Disclaimer:** this software is in **going-on development**. Some features can
change between different commits. A stable version is planned to be released
soon. The release v.0.0.1 corresponds to the date of submission of the
publication but in the meanwhile important changes are being done to facilitate
the use of the package.

If you find a problem when use it or if you want to provide us feedback, please
[open an issue](https://github.com/aramis-lab/ad-dl/issues).

## Getting Started
> Full instructions for installation and additional information can be found in
the [user documentation](http://www.clinica.run/clinicadl).

`clinicadl` currently supports macOS and Linux.

We recommend to use `conda` or `virtualenv` to create an environment and install
inside clinicadl:

```{.sourceCode .bash}
conda create --name clinicadl_env python=3.6
conda activate clinicadl_env
pip install clinicadl
```

## Overview

### How to use `clinicadl` ?

`clinicadl` is an utility to be used with the command line.

There are six kind of tasks that can be performed using the command line:

- **Process TSV files**. `tsvtool` includes many functions to get labels from
  BIDS, perform k-fold or single splits, produce demographic analysis of
  extracted labels and reproduce the restrictions made on AIBL and OASIS in the
  original paper.

- **Generate a synthetic dataset.** The `generate` task is useful to obtain
  synthetic datasets frequently used in functional tests.

- **T1w-weighted images preprocessing.** The `preprocessing` task processes a dataset of T1
  images stored in BIDS format and prepares to extract the tensors (see paper
  for details on the preprocessing). Output is stored using the
  [CAPS](http://www.clinica.run/doc/CAPS/Introduction/) hierarchy.

- **T1 MRI tensor extraction.** The `extract` task allows to create files in
  PyTorch format (`.pt`) with different options: the complete MRI, 2D slices
  and/or 3D patches. This files are also stored in the CAPS hierarchy.

- **Train neural networks.** The `train` task is designed to perform training
  of CNN models using different kind of inputs, e.g., a full MRI (3D-image),
  patches from a MRI (3D-patch), specific regions of a MRI (ROI-based) or
  slices extracted from the MRI (2D-slices). Parameters used during the
  training are configurable. This task allow also to train autoencoders.

- **MRI classification.** The `classify` task uses previously trained models
  to perform the inference of a particular or a set of MRI.

For detailed instructions and options of each task type  `clinica 'task' -h`.

## Testing

Be sure to have the `pytest` library in order to run the test suite.  This test
suite includes unit testing to be launched using the command line.

### Unit testing (WIP)

The CLI (command line interface) part is tested using `pytest`. We are planning
to provide unit tests for the other tasks in the future. If you want to run
successfully the tests maybe you can use a command like this one:

```{.sourceCode .bash}
pytest clinicadl/tests/test_cli.py
```

### Functional testing

Training task are tested using synthetic data created from MRI extracted of the
OASIS dataset. To run them, go to the test folder and type the following
command in the terminal:

```{.sourceCode .bash}
pytest ./test_train_cnn.py
```
Please, be sure to previously create the right dataset.

### Model prediction tests

For sanity check trivial datasets can be generated to train or test/validate
the predictive models.

The follow command allow you to generate two kinds of synthetic datasets: fully
separable (trivial) or intractable data (IRM with random noise added).

```{.sourceCode .bash}
clinicadl generate {random,trivial} caps_directory tsv_path output_directory
--n_subjects N_SUBJECTS
```
The intractable dataset will be made of noisy versions of the first image of
the tsv file given at
`tsv_path` associated to random labels.

The trivial dataset includes two labels:
- AD corresponding to images with the left half of the brain with lower
  intensities,
- CN corresponding to images with the right half of the brain with lower
  intensities.

## Pretrained models

Some of the pretained model for the CNN networks can be obtained here:
<https://zenodo.org/record/3491003>  

These models were obtained during the experiments for publication.
Updated versions of the models will be published soon.

## Bibliography

All the papers described in the State of the art section of the manuscript may
be found at this URL address: <https://www.zotero.org/groups/2337160/ad-dl>.

## Related Repositories

- [Clinica: Software platform for clinical neuroimaging studies](https://github.com/aramis-lab/clinica)
- [AD-ML: Framework for the reproducible classification of Alzheimer's disease using machine learning](https://github.com/aramis-lab/AD-ML)
