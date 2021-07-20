# `clinicadl predict` - Inference using pretrained models

This functionality performs individual prediction and metrics computation on a set of data using models trained with
[`clinicadl train`](./Train/Introduction.md) or [`clinicadl random-search generate`](./RandomSearch.md)
tasks. It can also use any pretrained models if they are structured like a [MAPS](./MAPS.md).

!!! warning
    For `patch`, `roi` and `slice` models, the predictions of the models on the
    validation set are needed to perform unbiased ensemble predictions at the image level. 
    If the tsv files in `fold-<fold>/best_<metric>/validation` were erased the task cannot
    be run.

## Prerequisites

In order to execute this task, the input images must be listed in a `tsv_file`.
Please check which preprocessing needs to
be performed in the `maps.json` file in the results folder. If it has
not been performed, execute the preprocessing pipeline as well as `clinicadl
extract` to obtain the tensor versions of the images.

<--!Some pretrained models are available to [download
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
clinicadl predict <caps_directory> <tsv_file> <model_path> <prefix_output>

```
where:

- `caps_directory` (str) is the input folder containing the neuroimaging data
  (tensor version of images, output of [`clinicadl extract`
  pipeline](Preprocessing/Extract.md)) in a
  [CAPS](https://aramislab.paris.inria.fr/clinica/docs/public/latest/CAPS/Introduction/) hierarchy.
- `tsv_file` (str) is a path to a TSV file with subjects/sessions to process (filename
  included) OR the path to the test folder of a split directory obtained with `clinicadl tsvtool split`.
- `model_path` (str) is a path to the folder where the model and the json file
  are stored.
- `prefix_output` (str) is a prefix to name the data set used for the prediction
  task.

Optional arguments:

- **Computational resources**
    - `--use_cpu` (bool) forces to use CPU. Default behaviour is to try to use a
      GPU and to raise an error if it is not found.
    - `--nproc` (int) is the number of workers used by the DataLoader. Default value: `2`.
    - `--batch_size` (int) is the size of the batch used in the DataLoader. Default value: `2`.
- **Other options**
    - `--no_labels` (bool) is a flag to add if the dataset does not contain ground truth labels. 
      Default behaviour will look for ground truth labels and raise an error if not found.
    - `--use_extracted_features` (bool) is a flag to use extracted slices or
      patches, if not specified they will be extracted on the fly from the complete
      image (if necessary). Default value: `False`.
    - `--selection_metrics` (list[str]) is a list of metrics to find the best models to evaluate.
      Default will predict the results for best model based on the loss only.
    - `--diagnoses` (list[str]) if `tsv_file` is a split directory, then will only load the labels wanted.
    Default will look for the same labels used during the training task.
    Choices available are `AD`, `CN`, `MCI`, `sMCI` and `pMCI`.
    - `--multi_cohort` (bool) is a flag indicated that [multi-cohort classification](Train/Details.md#multi-cohort)
     is performed.
    In this case, `caps_directory` and `tsv_path` must be paths to TSV files.

## Outputs

Results are stored in the results folder given by `model_path`, according to
the following file system:
```
<model_path>
    ├── fold-0  
    ├── ...  
    └── fold-<fold>
        └── best-<metric>
                └── prefix
                    ├── description.log
                    ├── <prefix_output>_image_level_metrics.tsv
                    ├── <prefix_output>_image_level_prediction.tsv
                    ├── <prefix_output>_{patch|roi|slice}_level_metrics.tsv
                    └── <prefix_output>_{patch|roi|slice}_level_prediction.tsv

```
The last two TSV files will be absent if the model takes as input the whole
image. Moreover, `*_metrics.tsv` files are not computed if `--no_labels` is given.
The content of `_prediction.tsv` files depend on the task performed during the training task.
