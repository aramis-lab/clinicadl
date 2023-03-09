# `interpret` - Interpret with attribution maps

This functionality allows interpreting pretrained models by computing a mean attribution map
across a group of images and optionally individual attribution maps. It takes as input MAPS-like model folders.

Two methods are currently implemented:
- `gradients` is an adaptation of [[Simonyan et al., 2014](https://arxiv.org/abs/1312.6034)],
- `grad-cam` is the implementation of [[Selvaraju et al., 2017](https://arxiv.org/abs/1610.02391v4)].


## Prerequisites

Please check which preprocessing needs to
be performed in the `maps.json` file of the MAPS. If it has
not been performed, execute the preprocessing pipeline as well as `clinicadl
extract` to obtain the tensor versions of the images.

<!--Some pretrained models are available to [download
here](https://aramislab.paris.inria.fr/clinicadl/files/models/). You
can download them using your navigator or the command line. For example, to get
the model "Image-based" with a single split type:

```
curl -k https://aramislab.paris.inria.fr/clinicadl/files/models/v1.1.0/maps_exp3.tar.gz -o maps_exp3.tar.gz
tar xf model_exp3_splits_1.tar.gz
```
-->

## Running the task
This task can be run with the following command line:
```Text
clinicadl interpret [OPTIONS] INPUT_MAPS_DIRECTORY DATA_GROUP NAME METHOD

```
where:

- `INPUT_MAPS_DIRECTORY` (Path) is a path to the MAPS folder containing the model which will be interpreted.
- `DATA_GROUP` (str) is a prefix to name the files resulting from the interpretation task.
- `NAME` (str) is the name of the saliency map task.
- `METHOD` (str) is the name of the saliency method (`gradients` or `grad-cam`).

!!! warning "data group consistency"
    For ClinicaDL, a data group is linked to a list of participants / sessions and a CAPS directory.
    When performing a prediction, interpretation or tensor serialization the user must give a data group.
    If this data group does not exist, the user MUST give a `caps_path` and a `tsv_path`.
    If this data group already exists, the user MUST not give any `caps_path` or `tsv_path`, or set overwrite to True.


Optional arguments:

- **Computational resources**
    - `--gpu / --no-gpu` (bool) Uses GPU acceleration or not. Default behaviour is to try to use a
      GPU. If not available an error is raised. Use the option `--no-gpu` if running in CPU.
    - `--n_proc` (int) is the number of workers used by the DataLoader. Default: `2`.
    - `--batch_size` (int) is the size of the batch used in the DataLoader. Default: `8`.
- **Model selection**
    - `--selection_metrics` (List[str]) is a list of metrics to find the best models to evaluate.
      Default will predict the results for best model based on the loss only.
- **Data management**
    - `--participants_tsv` (Path) is a path to a directory containing one TSV file per diagnosis
    (see output tree of [get-labels](./TSVTools.md#getlabels---extract-labels-specific-to-alzheimers-disease)). 
    Default will use the same participants as those used during the training task.
    - `--caps_directory` (Path) is the path to a [CAPS](https://aramislab.paris.inria.fr/clinica/docs/public/latest/CAPS/Introduction/) hierarchy.
    Default will use the same CAPS as during the training task.
    - `--multi_cohort` (bool) is a flag indicated that [multi-cohort classification](Train/Details.md#multi-cohort)
     is performed.
    In this case, `caps_directory` and `tsv_path` must be paths to TSV files.
    - `--diagnoses` (List[str]) if `tsv_file` is a split directory, then will only load the labels wanted.
    Default will look for the same labels used during the training task.
- **Other options**
    - `--target_node` (int) is the node the gradients explain. By default, it will target the first output node.
    - `--save_individual` (bool) is an option to save individual saliency maps in addition to the mean saliency map.
    - `--level_grad_cam` (int) is the layer considered to compute the Grad-CAM map. Default will use the last
    layer of the `convolutions` parameter of the targeted `CNN`. The minimum value `1` will backpropagate the results
    until the feature map located after the first layer.
   

## Outputs

Results for the `DATA_GROUP` level are stored in the results folder given by `INPUT_MAPS_DIRECTORY`, according to
the following file system:
```
<maps_directory>
    ├── split-0  
    ├── ...  
    └── split-<split>
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
  
