<h1 align="center">
  <a href="http://www.clinica.run">
    <img src="http://www.clinica.run/assets/images/clinica-icon-257x257.png" alt="Clinica Logo" width="120" height="120">
  </a>
  +
  <a href="https://pytorch.org/">
    <img src="https://pytorch.org/assets/images/pytorch-logo.png" alt="PyTorch Logo" width="120" height="120">
  </a>
  <br/>
  ClinicaDL
</h1>

<p align="center"><strong>Framework for the reproducible classification of Alzheimer's disease using deep learning</strong></p>

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
  <a href="https://clinicadl.readthedocs.io/">Documentation</a> |
  <a href="https://aramislab.paris.inria.fr/clinicadl/tuto/intro.html">Tutorial</a> |
  <a href="https://groups.google.com/forum/#!forum/clinica-user">Forum</a> |
  See also:
  <a href="#related-repositories">AD-ML</a>,
  <a href="#related-repositories">Clinica</a>
</p>


## About the project

This repository hosts the source code of a **framework for the reproducible
evaluation of deep learning classification experiments using anatomical MRI
data for the computer-aided diagnosis of Alzheimer's disease (AD)**. This work
has been published in [Medical Image
Analysis](https://doi.org/10.1016/j.media.2020.101694) and is also available on
[arXiv](https://arxiv.org/abs/1904.07773).

Automatic classification of AD using classical machine learning approaches can
be performed using the framework available here:
<https://github.com/aramis-lab/AD-ML>.

> **Disclaimer:** this software is **under development**. Some features can
change between different commits. A stable version is planned to be released
soon. The release v.0.0.1 corresponds to the date of submission of the
publication but in the meantime important changes are being done to facilitate
the use of the package.

The complete documentation of the project can be found on 
this [page](https://clinicadl.readthedocs.io/). 
If you find a problem when using it or if you want to provide us feedback,
please [open an issue](https://github.com/aramis-lab/ad-dl/issues) or write on
the [forum](https://groups.google.com/forum/#!forum/clinica-user).

## Getting started
ClinicaDL currently supports macOS and Linux.

We recommend to use `conda` or `virtualenv` for the installation of ClinicaDL
as it guarantees the correct management of libraries depending on common
packages:

```{.sourceCode .bash}
conda create --name ClinicaDL python=3.7
conda activate ClinicaDL
pip install clinicadl
```

:warning: **NEW!:** :warning:
> :reminder_ribbon: Visit our [hands-on tutorial web
site](https://aramislab.paris.inria.fr/clinicadl/tuto/intro.html) to start
using **ClinicaDL** directly in a Google Colab instance!

## Overview

### How to use ClinicaDL?

`clinicadl` is an utility that is used through the command line. Several tasks
can be performed:

- **Preparation of your imaging data**
    * **T1w-weighted MR image preprocessing.** The `preprocessing` task
      processes a dataset of T1 images stored in BIDS format and prepares to
      extract the tensors (see paper for details on the preprocessing). Output
      is stored using the [CAPS](http://www.clinica.run/doc/CAPS/Introduction/)
      hierarchy.
    * **Quality check of preprocessed data.** The `quality_check` task uses a
      pretrained network [(Fonov et al,
      2018)](https://www.biorxiv.org/content/10.1101/303487v1) to classify
      adequately registered images.
    * **Tensor extraction from preprocessed data.** The `extract` task allows
      to create files in PyTorch format (`.pt`) with different options: the
      complete MRI, 2D slices and/or 3D patches. This files are also stored in
      the [CAPS](http://www.clinica.run/doc/CAPS/Introduction/) hierarchy.

- **Train & test your classifier**
    * **Train neural networks.** The `train` task is designed to perform
      training of CNN models using different kind of inputs, e.g., a full MRI
      (3D-image), patches from a MRI (3D-patch), specific regions of a MRI
      (ROI-based) or slices extracted from the MRI (2D-slices). Parameters used
      during the training are configurable. This task allow also to train
      autoencoders.
    * **MRI classification.** The `classify` task uses previously trained models
      to perform the inference of a particular or a set of MRI.


- **Utilitaries used for the preparation of imaging data and/or training your
  classifier**
    * **Process TSV files**. `tsvtool` includes many functions to get labels
      from BIDS, perform k-fold or single splits, produce demographic analysis
      of extracted labels and reproduce the restrictions made on AIBL and OASIS
      in the original paper.
    * **Generate a synthetic dataset.** The `generate` task is useful to obtain
      synthetic datasets frequently used in functional tests.

## Pretrained models

Some of the pretained models for the CNN networks described in 
([Wen et al., 2020](https://doi.org/10.1016/j.media.2020.101694)) 
are available on Zenodo:
<https://zenodo.org/record/3491003>

Updated versions of the models will be published soon.

## Related Repositories

- [Clinica: Software platform for clinical neuroimaging studies](https://github.com/aramis-lab/clinica)
- [AD-ML: Framework for the reproducible classification of Alzheimer's disease using machine learning](https://github.com/aramis-lab/AD-ML)
