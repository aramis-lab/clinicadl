# `train` - Custom experiments

The aim of `clinicadl` is not only to provide a collection of tools, 
but also to allow users to add their own in the framework.
Before starting, please fork and clone the [github repo](https://github.com/aramis-lab/AD-DL]).

!!! tip
    Do not hesitate to ask for help on [GitHub](https://github.com/aramis-lab/AD-DL/issues/new) 
    or propose a new pull request!


## Custom architecture 

Custom CNN architectures can be added to clinicadl by adding a model class in `clinicadl/tools/deep_learning/models` 
and importing it in `clinicadl/tools/deep_learning/models/__init__.py`.

There are two rules to follow to convert this CNN architecture into an autoencoder:

1. Implement the convolutional part in `features` and the fully-connected layer in `classifier`. See predefined models as examples.
2. Check that all the modules used in your architecture are in the [list of modules](./Introduction.md#autoencoders-construction-from-cnn-architectures)
 transposed by the autoencoder or that the invert version of this module is itself (it is the case for activation layers).

## Custom input type

Input types that are already provided in `clinicadl` are `image`, `patch`, `roi` and `slice`. To add a custom input type, 
please follow the steps detailed below:

- Choose a mode name for this input type (for example default ones are image, patch, roi and slice). 
- Add your dataset class in `clinicadl/tools/deep_learning/data.py` as a child class of the abstract class `MRIDataset`.
- Create your dataset in `return_dataset` by adding:
```
elif mode==<mode_name>:
    return <dataset_class>(
        input_dir,
        data_df,
        preprocessing=preprocessing,
        transformations=transformations,
        <custom_args>
    )
```
- Add your custom subparser to `train` and complete `train_func` in `clinicadl/cli.py`.

## Custom preprocessing
Define the path of your new preprocessing in the `_get_path` method of `MRIDataset` in `clinicadl/tools/deep_learning/data.py`. 

You will also have to add the name of your preprocessing pipeline in the general command line by modifying the possible choices 
of the `preprocessing` argument of `train_pos_group` in `cli.py`.

## Custom labels
You can launch a classification task with clinicadl using any label. 
The input tsv files must include the columns `participant_id`, `session_id` and the name of your label column. 

The label will be associated to a class in the dictionary `label_code`. This code is saved in the
commandline.json file and will be reloaded to perform classification using other ClinicaDL functionnalities
(`classify` or `interpret`).

!!! warning "Classes representation in train and validation sets"
    If `network_type` is set to `classification`, all classes must be represented in both the training and validation sets as the number of 
    classes is deduced from the data. To ensure its representation in both sets, use the flag
    `--categorical_split_variable` in `split` or `kfold` functions.

