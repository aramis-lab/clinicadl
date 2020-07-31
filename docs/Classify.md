# `clinicadl classify` - Inference using pretrained models

This pipeline performs image classification using models trained using the [`clinicadl train`](./Train/Introduction.md) pipeline. 
Pretrained models can be also loaded (see appropriate section).
The classification task is performed on the best model according to the validation balanced accuracy 
for all the fold directories existing in the model folder.

## Prerequisites
This pipeline can only be applied to the output file system of `clinicadl train`. 
More specifically, it relies on the `commandline.json` file at the root of the result folder and will load the model(s) 
that led to the best validation balanced accuracy (.pth.tar file).

!!! warning
    For `patch`, `roi` and `slice` models, the predictions of the models on the validation set 
    are needed to perform unbiased soft-voting and find the prediction on the image level. 
    If the tsv files in `cnn_classification/best_balanced_accuracy` were erased the pipeline cannot be run.

The pipeline also needs the images listed in the input `tsv_file` in a CAPS-formatted dataset. 
Please check which preprocessing needs to be performed in the `commandline.json` file in the results folder. 
If it has not been performed, execute the preprocessing pipeline as well as `clinicadl extract` 
to obtain the tensor versions of the images.

## Running the pipeline
The pipeline can be run with the following command line:
```Text
clinicadl classify <caps_directory> <tsv_file> <model_path> 

```
where:

- `caps_directory` (str) is the input folder containing the neuroimaging data (tensor version of images, 
output of [`clinicadl extract` pipeline](./Extract.md)) in a [CAPS](http://www.clinica.run/doc/CAPS/Introduction/) hierarchy.
- `tsv_file` (str) is a TSV file with subjects/sessions to process (filename included).
- `model_path` (str) is a path to the folder where the model and the json file are stored.

Optional arguments:

- `--prefix_output` (str) is a prefix in the output filenames. Default value: `prefix_DB`.
- `--use_extracted_features` (bool) is a flag to use extracted slices or patches, 
if not specified they will be extracted on the fly from the complete image (if necessary). Default value: `False`.
- `--use_cpu` (bool) forces to use CPU. Default behaviour is to try to use a GPU and to raise an error if it is not found.

## Outputs

Results are stored in the results folder given by `model_path`, according to the following file system:
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
The last two TSV files will be absent if the model takes as input the whole image.
