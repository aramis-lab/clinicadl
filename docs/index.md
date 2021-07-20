# `clinicadl` Documentation

!!! tip "ClinicaDL tutorial"
    Visit our [hands-on tutorial web site](https://aramislab.paris.inria.fr/clinicadl/tuto/intro.html) 
    to try **ClinicaDL** directly in a Google Colab instance!

## Installation

See [Installation](./Installation.md) section for detailed instructions.

`clinicadl` can be installed on Mac OS X and Linux machines, and possibly on Windows computers with a Linux Virtual Machine.

We assume that users installing and using `clinicadl` are comfortable using the command line.

## User documentation (`clinicadl`)

### Prepare your imaging data
- `clinicadl preprocessing run` - [Preprocessing pipelines](Preprocessing/Introduction.md)
    - `t1-linear` - [Linear processing of T1w MR images](Preprocessing/T1_Linear.md): affine registration to the MNI standard space
    - `t1-extensive` - ['Extensive' processing of T1w MR images](Preprocessing/T1_Extensive.md): non linear registration to the MNI standard space
- `clinicadl preprocessing quality-check` - [Quality control of preprocessed data](Preprocessing/QualityCheck.md): use a pretrained network [[Fonov et al., 2018](https://www.biorxiv.org/content/10.1101/303487v1)] to classify adequately registered images.
- `clinicadl preprocessing extract-tensor` - [Prepare input data for deep learning with PyTorch](Preprocessing/Extract.md)


### Train & test your classifier
- `clinicadl random-search` - [Explore hyperparameters space by training random models](./RandomSearch.md)
- `clinicadl train` - [Train with your data and create a model](./Train/Introduction.md)
- `clinicadl classify` - [Classify one image or a list of images with your previously trained CNN](./Classify.md)
- `clinicadl interpret`- [Interpret trained CNNs on individual or group of images](./Interpret.md)

### Utilitaries <!--used for the preparation of imaging data and/or training your classifier-->

- `clinicadl generate` - [Generate synthetic data for functional tests](./Generate.md)
- `clinicadl tsvtool` - [Handle TSV files for metadata processing and data splits](./TSVTools.md)


[//]:## Pretrained models

[//]:Pretrained models for the CNN networks implemented in ClinicaDL can be obtained here:
[//]:<https://zenodo.org/record/3491003>  

[//]:These models were obtained during the experiments for publication.
[//]:They correspond to a previous version of ClinicaDL, hence their file system is not compatible with the current version.
[//]:Updated versions of most representative models are available [here](https://aramislab.paris.inria.fr/files/data/models/dl/models_v002/).

## Support
- [Report an issue on GitHub](https://github.com/aramis-lab/clinicadl/issues)
- Use the [`clinicadl` Google Group](https://groups.google.com/forum/#!forum/clinica-user) to ask for help!

## Contributions
If you want to contribute but are not familiar with GitHub, please read the [Contribute](./Contribute.md) section.
You will also find how to run the test suite to check that your modifications are ready to be integrated.

## License
`clinicadl` is distributed under the terms of the MIT license given [here](https://github.com/aramis-lab/clinicadl/blob/dev/LICENSE.txt).

## Citing `clinicadl`
For publications or communications using `clinicadl`, please cite [[Wen et al., 2020](https://doi.org/10.1016/j.media.2020.101694)] 
as well as the references mentioned on the wiki page of the pipelines you used 
(for example, citing PyTorch when using the `extract` pipeline).

!!! info "Disclaimer"
    `clinicadl` is a software for research studies. It is not intended for use in medical routine.

---

![Clinica_Partners_Banner](https://aramislab.paris.inria.fr/clinica/docs/public/latest/img/Clinica_Partners_Banner.png)
