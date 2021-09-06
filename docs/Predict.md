# `predict` - Inference using pretrained models

This functionality performs individual prediction and metrics computation on a set of data using models trained with
[`clinicadl train`](./Train/Introduction.md) or [`clinicadl random-search generate`](./RandomSearch.md)
tasks. It can also use any pretrained models if they are structured like a [MAPS](./Introduction.md).

!!! warning "unbiased image-level results"
    For `patch`, `roi` and `slice` models, the predictions of the models on the
    validation set are needed to perform unbiased ensemble predictions at the image level. 
    If the tsv files in `fold-<fold>/best-<metric>/validation` were erased the task cannot
    be run.

## Prerequisites

Please check which preprocessing needs to
be performed in the `maps.json` file in the results folder. If it has
not been performed, execute the preprocessing pipeline as well as `clinicadl
extract` to obtain the tensor versions of the images.

<!--Some pretrained models are available to [download
here](https://aramislab.paris.inria.fr/files/data/models/dl/models_v002/). You
can download them using your navigator or the command line. For example, to get
the model "Image-based" with a single split type:

```
curl -k https://aramislab.paris.inria.fr/files/data/models/dl/models_v002/model_exp3_splits_1.tar.gz  -o model_exp3_splits_1.tar.gz
tar xf model_exp3_splits_1.tar.gz
```
-->

## Running the task
This task can be run with the following command line:
```Text
clinicadl predict INPUT_MAPS_DIRECTORY DATA_GROUP
```
where:

- `INPUT_MAPS_DIRECTORY` (path) is the path to the MAPS of the pretrained model.
- `DATA_GROUP` (str) is the name of the data group used for the prediction.

!!! warning "data group consistency"
    For ClinicaDL, a data group is linked to a list of participants / sessions and a CAPS directory.
    When performing a prediction, interpretation or tensor serialization the user must give a data group.
    If this data group does not exist, the user MUST give a `caps_directory` and a `participants_tsv`.
    If this data group already exists, the user MUST not give any `caps_directory` or `participants_tsv`, or set overwrite to True.

Optional arguments:

- **Computational resources**
    - `--gpu / --no-gpu` (bool) Uses GPU acceleration or not. Default behaviour is to try to use a
      GPU. If not available an error is raised. Use the option `--no-gpu` if running in CPU.
    - `--n_proc` (int) is the number of workers used by the DataLoader. Default: `2`.
    - `--batch_size` (int) is the size of the batch used in the DataLoader. Default: `2`.
- **Other options**
    - `--caps_directory` (path) is the input folder containing the neuroimaging data
      (tensor version of images, output of [`clinicadl extract`
      pipeline](Preprocessing/Extract.md)) in a
      [CAPS](https://aramislab.paris.inria.fr/clinica/docs/public/latest/CAPS/Introduction/) hierarchy.
    - `--participants_tsv` (path) is a path to a TSV file with subjects/sessions to process (filename
      included), OR the path to the test folder of a split directory obtained with `clinicadl tsvtool split`.
    - `--labels/--no_labels` (bool) is a flag to add if the dataset does not contain ground truth labels. 
      Default behaviour will look for ground truth labels and raise an error if not found.
    - `--selection_metrics` (List[str]) is a list of metrics to find the best models to evaluate.
      Default will predict the results for best model based on the loss only.
    - `--diagnoses` (List[str]) if `tsv_file` is a split directory, then will only load the labels wanted.
    Default will look for the same labels used during the training task.
    - `--multi_cohort` (bool) is a flag indicated that [multi-cohort classification](Train/Details.md#multi-cohort)
     is performed.
    In this case, `caps_directory` and `tsv_path` must be paths to TSV files.
    - `--overwrite` (bool) is a flag allowing to overwrite a data group to redefine it. All results obtained
    for this data group will be erased.

## Outputs

Results are stored in the MAPS of path `model_path`, according to
the following file system:
```
<model_path>
    ├── fold-0  
    ├── ...  
    └── fold-<i>
        └── best-<metric>
                └── <data_group>
                    ├── description.log
                    ├── <prefix>_image_level_metrics.tsv
                    ├── <prefix>_image_level_prediction.tsv
                    ├── <prefix>_{patch|roi|slice}_level_metrics.tsv
                    └── <prefix>_{patch|roi|slice}_level_prediction.tsv

```
The last two TSV files will be absent if the model takes as input the whole
image. Moreover, `*_metrics.tsv` files are not computed if `--no_labels` is given.
The content of `*_prediction.tsv` files depend on the task performed during the training task.
