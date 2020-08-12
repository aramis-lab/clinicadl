# `clinicadl classify` - Inference using pretrained models

This functionality performs image classification using models trained with
[`clinicadl train`](./Train/Introduction.md) task. It can also use pretrained
models if their folder structure is similar to the struture created by the
command `clinicadl train`.  At the top level of each model folder there are two
files:

- `environment.txt` describes the Python and Pytorch version used during the
  training.
- `commandline.json` describes the training parameters used to create the
  model.

This two files allow `clinicadl` to load the model(s) that led to the best
validation balanced accuracy (.pth.tar file).

!!! warning
    For `patch`, `roi` and `slice` models, the predictions of the models on the
    validation set are needed to perform unbiased soft-voting and find the
    prediction on the image level.  If the tsv files in
    `cnn_classification/best_balanced_accuracy` were erased the pipeline cannot
    be run.

## Prerequisites

In order to execute this task, the input images must be listed in a `tsv_file`
formatted using the CAPS definition. Please check which preprocessing needs to
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
clinicadl classify <caps_directory> <tsv_file> <model_path> 

```
where:

- `caps_directory` (str) is the input folder containing the neuroimaging data
  (tensor version of images, output of [`clinicadl extract`
  pipeline](./Extract.md)) in a
  [CAPS](http://www.clinica.run/doc/CAPS/Introduction/) hierarchy.
- `tsv_file` (str) is a TSV file with subjects/sessions to process (filename
  included).
- `model_path` (str) is a path to the folder where the model and the json file
  are stored.

Optional arguments:

- `--prefix_output` (str) is a prefix in the output filenames. Default value:
  `prefix_DB`.
- `--use_extracted_features` (bool) is a flag to use extracted slices or
  patches, if not specified they will be extracted on the fly from the complete
  image (if necessary). Default value: `False`.
- `--use_cpu` (bool) forces to use CPU. Default behaviour is to try to use a
  GPU and to raise an error if it is not found.

## Outputs

Results are stored in the results folder given by `model_path`, according to
the following file system:
```
<model_path>
    ├── fold-0  
    ├── ...  
    └── fold-i  
        └── cnn_classification
                └── best_balanced_accuracy
                    ├── <prefix_output>_image_level_metrics.tsv
                    ├── <prefix_output>_image_level_prediction.tsv
                    ├── <prefix_output>_{patch|roi|slice}_level_metrics.tsv
                    └── <prefix_output>_{patch|roi|slice}_level_prediction.tsv

```
The last two TSV files will be absent if the model takes as input the whole
image.
