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
clinicadl interpret <model_path> <name>

```
where:

- `model_path` (str) is the path to the MAPS of the pretrained model.
- `name` (str) is the name of the saliency map task.

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
    - `--caps_dir` (str) is the path to a [CAPS](https://aramislab.paris.inria.fr/clinica/docs/public/latest/CAPS/Introduction/) hierarchy.
    Default will use the same CAPS than during the training task.
    - `--diagnosis` (str) is the diagnosis that will be loaded in `tsv_path`. Default value: `AD`.
    - `--target_diagnosis` (str) is the class the gradients explain. Default will explain
    the given diaggnosis.
    - `--baseline` (bool) is a flag to load only `_baseline.tsv` files instead of `.tsv` files comprising all the sessions. Default: `False`.
    - `--keep_true` (bool) allows to choose only the images correctly (`True`) or badly (`False`)
    classified by the CNN. Default will not perform any selection.
    - `--nifti_template_path` (str) is a path to a nifti template to retrieve the affine values
    needed to write Nifti files for 3D saliency maps. Default will use the identity matrix for the affine.
    - `--multi_cohort` (bool) is a flag indicated that [multi-cohort interpretation](Train/Details.md#multi-cohort) is performed.
    In this case, `caps_dir` and `tsv_path` must be paths to TSV files. If no new `caps_dir` and `tsv_path` are 
    given this argument is not taken into account. 
- **Results**
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
                └── intepretation
                    └── <name>
                        ├── description.log
                        ├── mean_<mode>-<k>_map.pt
                        └── sub-<i>_ses-<j>_<mode>-<k>.pt
```
- `description.log` is a text file describing the options used to interpret the network,
- `mean_<mode>-<k>_map.pt` is the tensor of the mean saliency map for mode `k` 
  across the data set used (always saved),
- `sub-<i>_ses-<j>_<mode>-<k>.pt` is the tensor of the saliency map for participant `i`, session `j`
  and mode_id `k` (saved only if flag `--save_individual` was given).
  