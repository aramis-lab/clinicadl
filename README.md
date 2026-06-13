<h1 align="center">
  <a href="https://clinicadl.readthedocs.io/en/stable/">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="docs/_static/logos/white_logo.png">
      <img src="docs/_static/logos/black_logo.png" alt="ClinicaDL Logo" width="120" height="120">
    </picture>
  </a>
  <br/>
  ClinicaDL
</h1>

<p align="center"><strong>Open-source Python library for reproducible deep learning in neuroimaging</strong></p>

<p align="center">
  <a href="https://pypi.org/project/clinicadl/">
    <img src="https://img.shields.io/pypi/v/clinicadl" alt="PyPI version">
  </a>
  <a href="https://img.shields.io/pypi/pyversions/clinicadl">
    <img src="https://img.shields.io/pypi/pyversions/clinicadl" alt="Python versions">
  </a>
  <a href='https://clinicadl.readthedocs.io/en/stable/'>
    <img src='https://readthedocs.org/projects/clinicadl/badge/?version=latest' alt='Documentation Status' />
  </a>
  <a href='https://github.com/aramis-lab/clinicadl/actions/workflows/test.yml'>
    <img src='https://github.com/aramis-lab/clinicadl/actions/workflows/test.yml/badge.svg' alt='CI tests status' />
  </a>
  <a href="https://codecov.io/gh/aramis-lab/clinicadl" > 
    <img src="https://codecov.io/gh/aramis-lab/clinicadl/graph/badge.svg?token=0FS4P8BWCJ"/> 
  </a>
  <a href='https://opensource.org/licenses/MIT'>
    <img src='https://img.shields.io/badge/License-MIT-yellow.svg' alt='License' />
  </a>
</p>


## About the project

ClinicaDL is a Python library to build end-to-end reproducible deep learning pipelines for neuroimaging studies. It works with data following the [BIDS](https://bids.neuroimaging.io/index.html) standard, or preprocessed outputs of [Clinica](https://aramislab.paris.inria.fr/clinica/docs/public/latest/).

It relies on the medical imaging frameworks [MONAI](https://project-monai.github.io/) and [TorchIO](https://docs.torchio.org/).

To access the full documentation of the project, follow [this link](https://clinicadl.readthedocs.io/en/stable/).

## Installation

See the [installation guidelines](https://clinicadl.readthedocs.io/en/stable/installation.html).

## Getting started

For a quick overview of ClinicaDL, read the [Quickstart section](https://clinicadl.readthedocs.io/en/stable/quickstart.html) of the documentation.

## Contributing

See the [contribution guidelines](https://clinicadl.readthedocs.io/en/stable/contributing.html).

## Related Repositories

- [Clinica: Software platform for clinical neuroimaging studies](https://github.com/aramis-lab/clinica)
- [AD-DL: Convolutional neural networks for classification of Alzheimer's disease: Overview and reproducible evaluation](https://github.com/aramis-lab/AD-DL)
- [AD-ML: Framework for the reproducible classification of Alzheimer's disease using machine learning](https://github.com/aramis-lab/AD-ML)

## Citing us

- Thibeau-Sutre, E., Díaz, M., Hassanaly, R., Routier, A., Dormont, D., Colliot, O., Burgos, N.: *ClinicaDL: an open-source deep learning software for reproducible neuroimaging processing*, 2021. [doi:10.1016/j.cmpb.2022.106818](https://doi.org/10.1016/j.cmpb.2022.106818) [Open Access version](https://inria.hal.science/hal-03351976)
- Routier, A., Burgos, N., Díaz, M., Bacci, M., Bottani, S., El-Rifai O., Fontanella, S., Gori, P., Guillon, J., Guyot, A., Hassanaly, R., Jacquemont, T.,  Lu, P., Marcoux, A.,  Moreau, T., Samper-González, J., Teichmann, M., Thibeau-Sutre, E., Vaillant G., Wen, J., Wild, A., Habert, M.-O., Durrleman, S., and Colliot, O.: *Clinica: An Open Source Software Platform for Reproducible Clinical Neuroscience Studies*, 2021. [doi:10.3389/fninf.2021.689675](https://doi.org/10.3389/fninf.2021.689675) [Open Access version](https://hal.inria.fr/hal-02308126)
