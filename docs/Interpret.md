# `clinicadl interpret` - Interpretation with saliency maps

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
clinicadl interpret <model_path> <data_group> <name>

```
where:

- `model_path` (str) is the path to the MAPS of the pretrained model.
- `data_group` (str) is the data is the name of the data group used for the interpretation.
- `name` (str) is the name of the saliency map task.

!!! warning "data group consistency"
    For ClinicaDL, a data group is linked to a list of participants / sessions and a CAPS directory.
    When performing a prediction, interpretation or tensor serialization the user must give a data group.
    If this data group does not exist, the user MUST give a `caps_path` and a `tsv_path`.
    If this data group already exists, the user MUST not give any `caps_path` or `tsv_path`, or set overwrite to True.


Optional arguments:

- **Computational resources**
    - `--use_cpu` (bool) forces to use CPU. Default behaviour is to try to use a
      GPU and to raise an error if it is not found.
    - `--nproc` (int) is the number of workers used by the DataLoader. Default value: `2`.
    - `--batch_size` (int) is the size of the batch used in the DataLoader. Default value: `2`.
- **Model selection**
    - `--selection_metrics` (list of str) corresponds to the metrics according to which the 
    [best models](Train/Details.md#model-selection) of `model_path` will be loaded. Default: `loss`.
- **Data management**
    - `--tsv_path` (str) is a path to a directory containing one TSV file per diagnosis
    (see output tree of [getlabels](./TSVTools.md#getlabels---extract-labels-specific-to-alzheimers-disease)). 
    Default will use the same participants than those used during the training task.
    - `--caps_directory` (str) is the path to a [CAPS](https://aramislab.paris.inria.fr/clinica/docs/public/latest/CAPS/Introduction/) hierarchy.
    Default will use the same CAPS than during the training task.
    - `--multi_cohort` (bool) is a flag indicated that [multi-cohort interpretation](Train/Details.md#multi-cohort) is performed.
    In this case, `caps_dir` and `tsv_path` must be paths to TSV files. If no new `caps_dir` and `tsv_path` are 
    given this argument is not taken into account. 
- **Results**
    - `--target_node` (str) is the class the gradients explain. Default will explain
    the given diaggnosis.
    - `--save_individual` (bool) if this flag is given the individual saliency maps of each input will be saved. 
      Default will only save the mean saliency map across the data set.
   

## Outputs

Results are stored in the MAPS of path `model_path`, according to
the following file system:
```
<model_path>
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
  