# `clinicadl interpret` - Intepretation with saliency maps

This functionality allows to intepret pretrained models by creating saliency maps 
based on single images or group of images. It uses models trained with
[`clinicadl train`](./Train/Introduction.md) task. It can also use pretrained
models if their folder structure is similar to the structure created by the
command `clinicadl train`.  At the top level of each model folder there are two
files:

- `environment.txt` is the result of `pip freeze` and describes the 
environment used during training.
- `commandline.json` describes the training parameters used to create the
  model.

The file `commandline.json` allows `clinicadl` to load the model(s) that led to the best
performance on the validation set according to one or several metrics (.pth.tar file).

!!! warning
    For `patch`, `roi` and `slice` models, the predictions of the models on the
    validation set are needed to perform unbiased soft-voting and find the
    prediction on the image level.  If the tsv files in
    `cnn_classification/best_<metric>` were erased the task cannot
    be run.


## Prerequisites

Please check which preprocessing needs to
be performed in the `commandline.json` file in the results folder.  If it has
not been performed, execute the preprocessing pipeline as well as `clinicadl
extract` to obtain the tensor versions of the images.

Some pretrained models are available to [download
here](https://aramislab.paris.inria.fr/files/data/models/dl/models_v002/). You
can download them using your navigator or the command line. For example, to get
the model "Image-based" with a single split type:

```
curl -k https://aramislab.paris.inria.fr/files/data/models/dl/models_v002/model_exp3_splits_1.tar.gz  -o model_exp3_splits_1.tar.gz
tar xf model_exp3_splits_1.tar.gz
```

## Running the task
This task can be run with the following command line:
```Text
clinicadl interpret <level> <model_path> <name>

```
where:

- `level` (str) indicates if only one saliency map is computed based on all images (`group`),
  or if one saliency map is computed for each image (`individual`).
- `model_path` (str) is a path to the folder where the model and the json file
  are stored.
- `name` (str) is the name of the saliency map task.

Optional arguments:

- **Computational resources**
    - `--use_cpu` (bool) forces to use CPU. Default behaviour is to try to use a
      GPU and to raise an error if it is not found.
    - `--nproc` (int) is the number of workers used by the DataLoader. Default value: `2`.
    - `--batch_size` (int) is the size of the batch used in the DataLoader. Default value: `2`.
- **Model selection**
    - `--selection` (list of str) corresponds to the metrics according to which the 
    [best models](Train/Details.md#model-selection) of `model_path` will be loaded. 
    Choices are `best_loss` and `best_balanced_accuracy`. Default: `best_balanced_accuracy`.
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
- **Results display**
    - `--vmax` (float) is the maximum value used for 2D saliency maps display. Default value: `0.5`.
   

## Outputs

Results for the `group` level are stored in the results folder given by `model_path`, according to
the following file system:
```
<model_path>
    ├── fold-0  
    ├── ...  
    └── fold-i  
        └── gradients
                └── <selection>
                    └── <name>
                        ├── data.tsv
                        ├── commandline.json
                        ├── map.nii.gz (3D images) | map.jpg (2D images)
                        └── map.npy

```
- `data.tsv` contains all the sessions used during the job,
- `commandline.json` is a file containing all the arguments necessary to reproduce the visualization,
- `map.npy` is the numpy array corresponding to the saliency maps and can be loaded with `numpy.load`.

The output tree for the `individual` level is quite similar, except that one occlusion is created
per session. Then the output tree also includes the `participant_id` and `session_id`.
