# ClinicaDL Documentation

## Installation

ClinicaDL can be installed on Mac OS X and Linux machines, and possibly on Windows computers with a Linux Virtual Machine.

We assume that users installing and using ClinicaDL are comfortable with using the command line.

- [Installation](./Installation)

## User documentation (`clinicadl`)

### Prepare your imaging data
- `clinicadl preprocessing` - [Preprocessing pipelines](Run/Introduction)
    - `t1-linear` - [Linear processing of T1w MR images](Run/T1_Linear): affine registration to the MNI standard space
    - `t1-extensive` - ['Extensive' processing of T1w MR images](Run/T1_Extensive): non linear registration to the MNI standard space
- `clinicadl extract` - [Prepare input data for deep learning with PyTorch](./Extract)


### Train & test your classifier
- `clinicadl train` - [Train with your data and create a model](/Train/Introduction)
- `clinicadl classify` - [Classify one image or a list of images with your previously trained model](./Classify)

### Utilitaries <!--used for the preparation of imaging data and/or training your classifier-->

- `clinicadl generate` - [Generate synthetic data for functional tests](./Generate)
- `clinicadl tsvtool` - [Handle TSV files for metadata processing and data splits](./TSVTools)


## Pretrained models

Some of the pretained model for the CNN networks can be obtained here:
<https://zenodo.org/record/3491003>  

These models were obtained during the experiments for publication.
Updated versions of the models will be published soon.

## Bibliography

All the papers described in the State of the art section of the manuscript may
be found at this URL address: <https://www.zotero.org/groups/2337160/ad-dl>.

## Support
- [Report an issue on GitHub](https://github.com/aramis-lab/AD-DL/issues)
- Use the [ClinicaDL Google Group](https://groups.google.com/forum/#!forum/clinica-user) to ask for help!

## License
ClinicaDL is distributed under the terms of the MIT license given [here](https://github.com/aramis-lab/AD-DL/blob/dev/LICENSE.txt).

## Citing ClinicaDL
For publications or communications using ClinicaDL, please cite ([Wen et al., 2020](https://doi.org/10.1016/j.media.2020.101694)) as well as the references mentioned on the wiki page of the pipelines you used (for example, citing PyTorch when using the `extract` pipeline).

!!! info "Disclaimer"
    ClinicaDL is a software for research studies. It is not intended for use in medical routine

---

![Clinica_Partners_Banner](http://www.clinica.run/doc/img/Clinica_Partners_Banner.png)
