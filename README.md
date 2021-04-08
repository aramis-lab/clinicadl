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
data for the computer-aided diagnosis of Alzheimer's disease (AD)**.

> **Disclaimer:** this software is **under development**. Some features can
change between different commits.

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

## Related Repositories

- [Clinica: Software platform for clinical neuroimaging studies](https://github.com/aramis-lab/clinica)
- [AD-ML: Framework for the reproducible classification of Alzheimer's disease using machine learning](https://github.com/aramis-lab/AD-ML)

## Reproducibility

- Wen et al., 2020 [Convolutional neural networks for classification of Alzheimer's disease: Overview and reproducible evaluation](https://doi.org/10.1016/j.media.2020.101694)
([arXiv version](https://arxiv.org/abs/1904.07773)). Corresponding version `v0.0.1`.