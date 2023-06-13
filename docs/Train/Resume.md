# `resume` - Resume a prematurely stopped job

This functionality allows to resume a prematurely stopped job trained with
[`clinicadl train`](Introduction.md) of [`clinicadl random-search`](../RandomSearch.md) tasks.
The files that are used by this function are the following:

- `maps.json` describes the training parameters used to create the
  model,
- `checkpoint.pth.tar` contains the last version of the weights of the network,
- `optimizer.pth.tar` contains the last version of the parameters of the optimizer,
- `training.tsv` contains the successive values of the metrics during training.

These files are organized in `model_path` using the [MAPS format](../Introduction.md).

You should also ensure that the data at `tsv_path` and `caps_dir` in `maps.json`
is still present and correspond to the ones used during training.

## Prerequisites

Please check which preprocessing needs to
be performed in the `maps.json` file in the results folder. If it has
not been performed, execute the preprocessing pipeline as well as `clinicadl
extract` to obtain the tensor versions of the images.

## Running the task
This task can be run with the following command line:
```Text
clinicadl train resume INPUT_MAPS_DIRECTORY

```
where `INPUT_MAPS_DIRECTORY` (Path) is a path to the [MAPS folder](../Introduction.md) of the model.

The splits that must be resumed can be specified with the option `--split`. Default will resume and train
all possible splits allowed by the validation setting.

## Outputs

The outputs are formatted according to the [MAPS](../Introduction.md).

!!! note
    The files `checkpoint.pth.tar` and `optimizer.pth.tar` are automatically removed as soon
    as the [stopping criterion](Details.md#stopping-criterion) is reached, and the 
    performances of the models are evaluated on the training and validation datasets.
