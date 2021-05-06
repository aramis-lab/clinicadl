# `clinicadl train resume` - Resume a prematurely stopped job

This functionality allows to resume a prematurely stopped job trained with
[`clinicadl train`](Introduction.md) of [`clinicadl random-search generate`](../RandomSearch.md) tasks.
The files that are used by this function are the following:

- `commandline.json` describes the training parameters used to create the
  model,
- `checkpoint.pth.tar` contains the last version of the weights of the network,
- `optimizer.pth.tar` contains the last version of the parameters of the optimizer,
- `training.tsv` contains the successive values of the metrics during training.

These files are organized in `model_path` as follows:

```
<model_path>
├── commandline.json
└── fold-<i>
    ├── models
    │   ├── best_balanced_accuracy
    │   │   └── model_best.pth.tar
    │   ├── best_loss
    │   │   └── model_best.pth.tar
    │   ├── checkpoint.pth.tar
    │   └── optimizer.pth.tar
    ├── tensorboard_logs
    │   ├── train
    │   │   └── events.out.tfevents.1616090758.r7i7n7
    │   └── validation
    │       └── events.out.tfevents.1616090758.r7i7n7
    └── training.tsv
```

You should also ensure that the data at `tsv_path` and `caps_dir` in `commandline.json`
is still present and correspond to the ones used during training.

## Prerequisites

Please check which preprocessing needs to
be performed in the `commandline.json` file in the results folder. If it has
not been performed, execute the preprocessing pipeline as well as `clinicadl
extract` to obtain the tensor versions of the images.

## Running the task
This task can be run with the following command line:
```Text
clinicadl train resume <model_path>

```
where `model_path` (str) is a path to the folder where the model and the json file
are stored.

By default the arguments corresponding to computational resources will be the same
than the ones defined in `commandline.json`. However it is possible to change them
by using the following options:

- `--nproc` (int) changes the number of workers used by the DataLoader.
- `--use_cpu` (bool) forces to use CPU.
- `--use_gpu` (bool) forces to use GPU.
- `--batch_size` (int) changes the size of the batch used in the DataLoader.
- `--evaluation_steps` (int) changes the number of iterations to perform before
computing an evaluation.

## Outputs

The outputs correspond to the ones obtained using [`clinicadl train`](Introduction.md#outputs)

!!! note
    The files `checkpoint.pth.tar` and `optimizer.pth.tar` are automatically removed as soon
    as the [stopping criterion](Details.md#stopping-criterion) is reached and the 
    performances of the models are evaluated on the training and validation datasets.
