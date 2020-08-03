# `generate` - Produce synthetic data for debugging & functional tests

This pipeline generates a synthetic dataset for a binary classification task from a CAPS-formatted dataset. 
It produces a new CAPS containing either `trivial` or `random` data:

- Trivial data should be perfectly classified by a classifier. Each label corresponds to images whose intensities of 
respectively the right or the left hemisphere are strongly decreased.
- Random data cannot be correctly classified. All the images from this dataset comes from the same image to which random noise is added. 
Then the images are randomly distributed between the two labels.

![Schemes of trivial and random data](./images/generate.png)

Both variants were used for functional testing of the final models proposed in 
[[Wen et al, 2020](https://www.sciencedirect.com/science/article/abs/pii/S1361841520300591)].
Moreover, trivial data are useful for debugging a framework: hyper parameters can be more easily tested as 
fewer data samples are required and convergence should be reached faster as the classification task is simpler.

## Prerequisites
You need to execute the `clinicadl preprocessing` and `clinicadl extract` pipelines prior to running this pipeline.
Future versions will include the possibility to perform `generate` on the tensors extracted from another preprocessing pipeline, 
`t1-extensive`.

!!! note
    The `trivial` option can synthesize at most a number of images per label that is equal to the total number of images 
    in the input CAPS , while the `random` option can synthesize as many images as wanted with only one input image.

## Running the pipeline
The pipeline can be run with the following command line:
```
clinicadl generate <dataset> <caps_directory> <tsv_path> <output_dir>
```
where:

- `dataset` (str) is the type of synthetic data wanted. Choices are `random` or `trivial`.
- `caps_directory` (str) is the input folder containing the neuroimaging data in a [CAPS](http://www.clinica.run/doc/CAPS/Introduction/) hierarchy.
- `tsv_path` (str) is the path to a tsv file containing the subjects/sessions list for data generation.
- `output_dir` (str) is the folder where the synthetic CAPS is stored.


Pipeline options:

- `--n_subjects` (int) number of subjects per label in the synthetic dataset. Default value: `300`.
- `--preprocessing` (str) preprocessing pipeline used in the input `caps_directory`. Must be `t1-linear` 
(t1-extensive to be added soon !). Default value: `t1-linear`.
- `--mean` (float) Specific to random. Mean value of the gaussian noise added to images. Default value: `0`.
- `--sigma` (float) Specific to random. Standard deviation of the gaussian noise added to images. Default value: `0.5`.
- `--mask_path` (str) Specific to trivial. Path to the atrophy masks used to generate the two labels. 
Default will download masks based on AAL2 in `clinicadl/resources/masks`.
- `--atrophy_percent` (float) Specific to trivial. Percentage of intensity decrease applied to the regions targeted by the masks. Default value: 60. 

!!! tip
    Do not hesitate to type `clinicadl generate --help` to see the full list of parameters.


## Outputs
Results are stored in the same folder hierarchy as the input folder. 
