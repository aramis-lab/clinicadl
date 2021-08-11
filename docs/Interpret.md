# `interpret` - Interpretation with gradient maps

This functionality allows interpreting pretrained models by computing a mean saliency map
across a group of images. It takes as input MAPS-like model folders.

## Prerequisites

Please check which preprocessing needs to
be performed in the `maps.json` file of the MAPS. If it has
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
clinicadl interpret INPUT_MAPS_DIRECTORY DATA_GROUP NAME

```
where:

- `INPUT_MAPS_DIRECTORY` (path) is a path to the MAPS folder where the model and the json file
  are stored.
- `DATA_GROUP` (str) is a prefix to name the files resulting from the interpretation task.
- `NAME` (str) is the name of the saliency map task.

!!! warning "data group consistency"
    For ClinicaDL, a data group is linked to a list of participants / sessions and a CAPS directory.
    When performing a prediction, interpretation or tensor serialization the user must give a data group.
    If this data group does not exist, the user MUST give a `caps_path` and a `tsv_path`.
    If this data group already exists, the user MUST not give any `caps_path` or `tsv_path`, or set overwrite to True.


Optional arguments:

- **Computational resources**
    - `--use_cpu` (bool) forces using CPUs. Default behaviour is to try to use a
      GPU and to raise an error if it is not found.
    - `--nproc` (int) is the number of workers used by the DataLoader. Default value: `2`.
    - `--batch_size` (int) is the size of the batch used in the DataLoader. Default value: `2`.
- **Model selection**
    - `--selection_metrics` (list of str) corresponds to the metrics according to which the 
    [best models](Train/Details.md#model-selection) of `INPUT_MAPS_DIRECTORY` will be loaded. 
    Choices are `best_loss` and `best_balanced_accuracy`. Default: `best_loss`.
- **Data management**
    - `--participants_tsv` (str) is a path to a directory containing one TSV file per diagnosis
    (see output tree of [getlabels](./TSVTools.md#getlabels---extract-labels-specific-to-alzheimers-disease)). 
    Default will use the same participants as those used during the training task.
    - `--caps_directory` (str) is the path to a [CAPS](https://aramislab.paris.inria.fr/clinica/docs/public/latest/CAPS/Introduction/) hierarchy.
    Default will use the same CAPS as during the training task.
    - `--diagnosis` (str) is the diagnosis that will be loaded in `participants_tsv`. Default value: `AD`.
    - `--target_diagnosis` (str) is the class the gradients explain. Default will explain
    the given diagnosis.
    - `--baseline` (bool) is a flag to load only `_baseline.tsv` files instead of `.tsv` files comprising all the sessions. Default: `False`.
    - `--keep_true` (bool) allows choosing only the images correctly (`True`) or badly (`False`)
    classified by the CNN. Default will not perform any selection.
    - `--nifti_template_path` (str) is a path to a nifti template to retrieve the affine values
    needed to write Nifti files for 3D saliency maps. Default will use the identity matrix for the affine.
    - `--multi_cohort` (bool) is a flag indicated that [multi-cohort interpretation](Train/Details.md#multi-cohort) is performed.
    In this case, `caps_directory` and `participants_tsv` must be paths to TSV files. If no new `caps_directory` and `participants_tsv` are 
    given this argument is not taken into account. 
- **Results display**
    - `--vmax` (float) is the maximum value used for 2D saliency maps display. Default value: `0.5`.
- **Other options**
    - `--target_node` (str) is the node the gradients explain. By default it will target the first output node.
    - `save_individual` (str) is an option to save individual saliency maps in addition to the mean saliency map.
   

## Outputs

Results for the `DATA_GROUP` level are stored in the results folder given by `INPUT_MAPS_DIRECTORY`, according to
the following file system:
```
<maps_directory>
    ├── fold-0  
    ├── ...  
    └── fold-<fold>
        └── best-<metric>
                └── <data_group>
                    └── interpret-<name>
                        ├── mean_<mode>-<k>_map.pt
                        └── sub-<i>_ses-<j>_<mode>-<k>.pt
```

- `mean_<mode>-<k>_map.pt` is the tensor of the mean saliency map for mode `k` 
  across the data set used (always saved),
- `sub-<i>_ses-<j>_<mode>-<k>.pt` is the tensor of the saliency map for participant `i`, session `j`
  and mode_id `k` (saved only if flag `--save_individual` was given).
  