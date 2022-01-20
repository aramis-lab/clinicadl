# `save-tensors` - Save reconstruction outputs

This tool allows to save the output tensors of a whole [data group](./Introduction.md), associated with the tensor
corresponding to their input.
This can be useful for the `reconstruction` task, for which the user may want to perform
extra analyses directly on the images reconstructed by a trained network, or simply visualize
them for a qualitative check.

## Prerequisites

Please check which preprocessing needs to
be performed in the `maps.json` file of the MAPS. If it has
not been performed, execute the preprocessing pipeline as well as `clinicadl
extract` to obtain the tensor versions of the images.

## Running the task
This task can be run with the following command line:
```Text
clinicadl save-tensor [OPTIONS] INPUT_MAPS_DIRECTORY DATA_GROUP

```
where:

- `INPUT_MAPS_DIRECTORY` (Path) is a path to the MAPS folder containing the model which will be interpreted.
- `DATA_GROUP` (str) is a prefix to name the files resulting from the interpretation task.

!!! warning "data group consistency"
    For ClinicaDL, a data group is linked to a list of participants / sessions and a CAPS directory.
    When performing a prediction, interpretation or tensor serialization the user must give a data group.
    If this data group does not exist, the user MUST give a `caps_path` and a `tsv_path`.
    If this data group already exists, the user MUST not give any `caps_path` or `tsv_path`, or set overwrite to True.


Optional arguments:

- **Computational resources**
    - `--gpu / --no-gpu` (bool) Uses GPU acceleration or not. Default behavior is to try to use a
      GPU. If not available an error is raised. Use the option `--no-gpu` if running in CPU.
    - `--n_proc` (int) is the number of workers used by the DataLoader. Default: `2`.
    - `--batch_size` (int) is the size of the batch used in the DataLoader. Default: `2`.
- **Model selection**
    - `--selection_metrics` (List[str]) is a list of metrics to find the best models to evaluate.
      Default will predict the results for best model based on the loss only.
- **Data management**
    - `--participants_tsv` (Path) is a path to a directory containing one TSV file per diagnosis
    (see output tree of [getlabels](./TSVTools.md#getlabels---extract-labels-specific-to-alzheimers-disease)). 
    Default will use the same participants as those used during the training task.
    - `--caps_directory` (Path) is the path to a [CAPS](https://aramislab.paris.inria.fr/clinica/docs/public/latest/CAPS/Introduction/) hierarchy.
    Default will use the same CAPS as during the training task.
    - `--multi_cohort` (bool) is a flag indicated that [multi-cohort classification](Train/Details.md#multi-cohort)
     is performed.
    In this case, `caps_directory` and `tsv_path` must be paths to TSV files.
    - `--diagnoses` (List[str]) if `tsv_file` is a split directory, then will only load the labels wanted.
    Default will look for the same labels used during the training task.

## Outputs

Results are stored in the MAPS of path `maps_directory`, according to
the following file system:
```
<maps_directory>
    ├── fold-0  
    ├── ...  
    └── fold-<fold>
        └── best-<metric>
                └── <data_group>
                    └── tensors
                        ├── <participant_id>_<session_id>_{image|patch|roi|slice}-<i>_input.pt
                        └── <participant_id>_<session_id>_{image|patch|roi|slice}-<i>_output.pt
```
For each `participant_id`, `session_id` and index of the part of the image (`i`),
the input and the output tensors are saved in.