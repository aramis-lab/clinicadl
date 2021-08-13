# `generate` - Generate synthetic data sets

This command generates a synthetic dataset for a binary classification task from a CAPS-formatted dataset. 
It produces a new CAPS containing either `trivial` or `random` data:

- Trivial data should be perfectly classified by a classifier. Each label corresponds to brain images whose intensities of 
respectively the right or left hemisphere are strongly decreased.
- Random data cannot be correctly classified. All the images from this dataset comes from the same image to which random noise is added. 
Then the images are randomly distributed between the two labels.

![Schemes of trivial and random data](../images/generate.png)

Both variants were used for functional testing of the final models proposed in 
[[Wen et al., 2020](https://www.sciencedirect.com/science/article/abs/pii/S1361841520300591)].
Moreover, trivial data are useful for debugging a framework: hyper parameters can be more easily tested as 
fewer data samples are required and convergence should be reached faster as the classification task is simpler.

## Prerequisites
You need to execute the `clinica run` and `clinicadl extract` pipelines prior to running this task.

!!! note
    The `trivial` option can synthesize at most a number of images per label that is equal to the total number of images 
    in the input CAPS , while the `random` option can synthesize as many images as wanted with only one input image.

## Running the task

### `trivial`
The task can be run with the following command line:
```
clinicadl generate trivial CAPS_DIRECTORY GENERATED_CAPS_DIRECTORY
```
where:

- `CAPS_DIRECTORY` (str) is the input folder containing the neuroimaging data in a [CAPS](https://aramislab.paris.inria.fr/clinica/docs/public/latest/CAPS/Introduction/) hierarchy.
- `GENERATED_CAPS_DIRECTORY` (str) is the folder where the synthetic CAPS is stored.

Options:

- `--participants_tsv` (str) is the path to a tsv file containing the subjects/sessions list for data generation.
- `--n_subjects` (int) number of subjects per label in the synthetic dataset. Default value: `300`.
- `--preprocessing` (str) preprocessing pipeline used in the input `caps_directory`. Default value: `t1-linear`.
- `--mask_path` (str) Path to the atrophy masks used to generate the two labels. 
Default will download masks based on AAL2 in `clinicadl/resources/masks`.
- `--atrophy_percent` (float) Percentage of intensity decrease applied to the regions targeted by the masks. Default value: 60. 


### `random`
The task can be run with the following command line:
```
clinicadl generate random CAPS_DIRECTORY GENERATED_CAPS_DIRECTORY
```
where:

- `CAPS_DIRECTORY` (str) is the input folder containing the neuroimaging data in a [CAPS](https://aramislab.paris.inria.fr/clinica/docs/public/latest/CAPS/Introduction/) hierarchy.
- `GENERATED_CAPS_DIRECTORY` (str) is the folder where the synthetic CAPS is stored.

Options:

- `--participants_tsv` (str) is the path to a tsv file containing the subjects/sessions list for data generation.
- `--n_subjects` (int) number of subjects per label in the synthetic dataset. Default value: `300`.
- `--preprocessing` (str) preprocessing pipeline used in the input `caps_directory`. Default value: `t1-linear`.
- `--mean` (float) Mean value of the gaussian noise added to images. Default value: `0`.
- `--sigma` (float) Standard deviation of the gaussian noise added to images. Default value: `0.5`.

!!! tip
    Do not hesitate to type `clinicadl generate --help` to see the full list of parameters.


## Outputs
Results are stored in the same folder hierarchy as the input folder. 
