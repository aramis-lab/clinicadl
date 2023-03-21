<h1 align="center">
  <a href="http://www.clinicadl.readthedocs.io">
    <img src="https://clinicadl.readthedocs.io/en/latest/images/logo.png" alt="ClinicaDL Logo" width="120" height="120">
  </a>
  <br/>
  ClinicaDL
</h1>

<p align="center"><strong>Framework for the reproducible processing of neuroimaging data with deep learning methods</strong></p>

<p align="center">
  <a href="https://ci.inria.fr/clinicadl/job/AD-DL/job/dev/">
    <img src="https://ci.inria.fr/clinicadl/buildStatus/icon?job=AD-DL%2Fdev" alt="Build Status">
  </a>
  <a href="https://badge.fury.io/py/clinicadl">
    <img src="https://badge.fury.io/py/clinicadl.svg" alt="PyPI version">
  </a>
  <a href='https://clinicadl.readthedocs.io/en/latest/?badge=latest'>
    <img src='https://readthedocs.org/projects/clinicadl/badge/?version=latest' alt='Documentation Status' />
  </a>
  <a href='https://pepy.tech/project/clinicadl'>
    <img src='https://static.pepy.tech/badge/clinicadl/month' alt='Downloads' />
  </a>
</p>

<p align="center">
  <a href="https://clinicadl.readthedocs.io/">Documentation</a> |
  <a href="https://aramislab.paris.inria.fr/clinicadl/tuto">Tutorial</a> |
  <a href="https://groups.google.com/forum/#!forum/clinica-user">Forum</a>
</p>


## About the project

This repository hosts ClinicaDL, the deep learning extension of [Clinica](https://github.com/aramis-lab/clinica), 
a python library to process neuroimaging data in [BIDS](https://bids.neuroimaging.io/index.html) format.

> **Disclaimer:** this software is **under development**. Some features can
change between different releases and/or commits.

To access the full documentation of the project, follow the link 
[https://clinicadl.readthedocs.io/](https://clinicadl.readthedocs.io/). 
If you find a problem when using it or if you want to provide us feedback,
please [open an issue](https://github.com/aramis-lab/ad-dl/issues) or write on
the [forum](https://groups.google.com/forum/#!forum/clinica-user).

## Getting started
ClinicaDL currently supports macOS and Linux.

We recommend to use `conda` or `virtualenv` for the installation of ClinicaDL
as it guarantees the correct management of libraries depending on common
packages:

```{.sourceCode .bash}
conda create --name ClinicaDL python=3.8
conda activate ClinicaDL
pip install clinicadl
```

## Tutorial 
Visit our [hands-on tutorial web
site](https://aramislab.paris.inria.fr/clinicadl/tuto) to start
using **ClinicaDL** directly in a Google Colab instance!

## Related Repositories

- [Clinica: Software platform for clinical neuroimaging studies](https://github.com/aramis-lab/clinica)
- [AD-DL: Convolutional neural networks for classification of Alzheimer's disease: Overview and reproducible evaluation](https://github.com/aramis-lab/AD-DL)
- [AD-ML: Framework for the reproducible classification of Alzheimer's disease using machine learning](https://github.com/aramis-lab/AD-ML)

## Citing us

- Thibeau-Sutre, E., Díaz, M., Hassanaly, R., Routier, A., Dormont, D., Colliot, O., Burgos, N.: ‘ClinicaDL: an open-source deep learning software for reproducible neuroimaging processing‘, 2021. [hal-03351976](https://hal.inria.fr/hal-03351976)
- Routier, A., Burgos, N., Díaz, M., Bacci, M., Bottani, S., El-Rifai O., Fontanella, S., Gori, P., Guillon, J., Guyot, A., Hassanaly, R., Jacquemont, T.,  Lu, P., Marcoux, A.,  Moreau, T., Samper-González, J., Teichmann, M., Thibeau-Sutre, E., Vaillant G., Wen, J., Wild, A., Habert, M.-O., Durrleman, S., and Colliot, O.: ‘Clinica: An Open Source Software Platform for Reproducible Clinical Neuroscience Studies’, 2021. [doi:10.3389/fninf.2021.689675](https://doi.org/10.3389/fninf.2021.689675) [Open Access version](https://hal.inria.fr/hal-02308126)
