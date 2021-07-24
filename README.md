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

<p align="center"><strong>Framework for the reproducible processing of neuroimaging data with deep learning methods</strong></p>

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
- [AD-DL: Convolutional neural networks for classification of Alzheimer's disease: Overview and reproducible evaluation](https://github.com/aramis-lab/AD-DL)
- [AD-ML: Framework for the reproducible classification of Alzheimer's disease using machine learning](https://github.com/aramis-lab/AD-ML)

## Citing us

- Wen, J., Thibeau-Sutre, E., Samper-González, J., Routier, A., Bottani, S., Durrleman, S., Burgos, N., and Colliot, O.: ‘Convolutional Neural Networks for Classification of Alzheimer’s Disease: Overview and Reproducible Evaluation’, *Medical Image Analysis*, 63: 101694, 2020. [doi:10.1016/j.media.2020.101694](https://doi.org/10.1016/j.media.2020.101694)
- Routier, A., Burgos, N., Díaz, M., Bacci, M., Bottani, S., El-Rifai O., Fontanella, S., Gori, P., Guillon, J., Guyot, A., Hassanaly, R., Jacquemont, T.,  Lu, P., Marcoux, A.,  Moreau, T., Samper-González, J., Teichmann, M., Thibeau-Sutre, E., Vaillant G., Wen, J., Wild, A., Habert, M.-O., Durrleman, S., and Colliot, O.: ‘Clinica: An Open Source Software Platform for Reproducible Clinical Neuroscience Studies’, 2021. [hal-02308126](https://hal.inria.fr/hal-02308126)
