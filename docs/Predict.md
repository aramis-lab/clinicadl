# `predict` - Inference using pretrained models

This functionality performs individual prediction and metrics computation on a set of data using models trained with
[`clinicadl train`](./Train/Introduction.md) or [`clinicadl random-search`](./RandomSearch.md)
tasks. It can also use any pretrained models if they are structured like a [MAPS](./Introduction.md).

!!! warning "unbiased image-level results"
    For `patch`, `roi` and `slice` models, the predictions of the models on the
    validation set are needed to perform unbiased ensemble predictions at the image level. 
    If the tsv files in `split-<i>/best-<metric>/validation` were erased the task cannot
    be run.

## Prerequisites

Please check which preprocessing needs to
be performed in the `maps.json` file in the results folder. If it has
not been performed, execute the preprocessing pipeline as well as `clinicadl
extract` to obtain the tensor versions of the images.

!!! tip "Pretrained models"
    Some pretrained model are available to download through your browser or the
    command line (using `curl` or `wget`) ([see this
    section](https://clinicadl.readthedocs.io/en/stable/#pretrained-models)).
    These models are in MAPS format.

## Running the task
This task can be run with the following command line:
```Text
clinicadl predict [OPTIONS] INPUT_MAPS_DIRECTORY DATA_GROUP
```
where:

- `INPUT_MAPS_DIRECTORY` (Path) is the path to the MAPS of the pretrained model.
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
    - `--batch_size` (int) is the size of the batch used in the DataLoader. Default: `8`.
- **Reconstruction**
This tool allows to save the output tensors of a whole [data group](./Introduction.md), associated with the tensor corresponding to their input.
This can be useful for the `reconstruction` task, for which the user may want to perform extra analyses directly on the images reconstructed by a trained network, or simply visualize them for a qualitative check.
    - `--save_tensor` (flag) to the reconstruction output in the MAPS in Pytorch tensor format.
    - `--save_nifti` (flag) to the reconstruction output in the MAPS in NIfTI format.
- **Other options**
    - `--caps_directory` (Path) is the input folder containing the neuroimaging data
      (tensor version of images, output of [`clinicadl prepare-data`
      pipeline](Preprocessing/Extract.md)) in a
      [CAPS](https://aramislab.paris.inria.fr/clinica/docs/public/latest/CAPS/Introduction/) hierarchy.
    - `--participants_tsv` (Path) is a path to a TSV file with subjects/sessions to process (filename
      included), OR the path to the test folder of a split directory obtained with `clinicadl tsvtools split`.
    - `--use_labels/--no_labels` (bool) is a flag to add if the dataset does not contain ground truth labels. 
      Default behaviour will look for ground truth labels and raise an error if not found.
    - `--selection_metrics` (List[str]) is a list of metrics to find the best models to evaluate.
      Default will predict the results for best model based on the loss only.
    - `--diagnoses` (List[str]) if `tsv_file` is a split directory, then will only load the labels wanted.
    Default will look for the same labels used during the training task.
    - `--multi_cohort` (bool) is a flag indicated that [multi-cohort classification](Train/Details.md#multi-cohort)
     is performed.
    In this case, `caps_directory` and `tsv_path` must be paths to TSV files.
    - `--label` (str) name of the target label used in classification and regression tasks.
      Default will reuse the same label as in the training task.
    - `--overwrite` (bool) is a flag allowing to overwrite a data group to redefine it. All results obtained
    for this data group will be erased.

## Outputs

Results are stored in the MAPS of path `model_path`, according to
the following file system:
```
<model_path>
    ├── split-0  
    ├── ...  
    └── split-<i>
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

Results for reconstruction `--save_tensor` and `--save_nifti` are stored in the MAPS of path `maps_directory`, according to the following file system:
```
<maps_directory>
    ├── split-0  
    ├── ...  
    └── split-<i>
        └── best-<metric>
                └── <data_group>
                    └── tensors
                        ├── <participant_id>_<session_id>_{image|patch|roi|slice}-<X>_input.pt
                        └── <participant_id>_<session_id>_{image|patch|roi|slice}-<X>_output.pt
                    └── nifti_images
                        ├── <participant_id>_<session_id>_image_input.nii.gz
                        └── <participant_id>_<session_id>_image_output.nii.gz
```
For each `participant_id`, `session_id` and index of the part of the image (`X`),
the input and the output tensors are saved in.
