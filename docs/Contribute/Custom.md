# `train` - Custom experiments

The aim of `clinicadl` is not only to provide a collection of tools, 
but also to allow users to add their own in the framework.
Before starting, please fork and clone the [github repo](https://github.com/aramis-lab/clinicadl]) or
read the [newcomers section](./Newcomers.md) if you are not familiar with GitHub.

!!! tip
    Do not hesitate to ask for help on [GitHub](https://github.com/aramis-lab/clinicadl/issues/new), 
    or to propose a new pull request!


## Custom architecture 

Custom architectures can be added to ClinicaDL by adding a model class in `clinicadl/utils/network` 
and importing it in `clinicadl/utils/network/__init__.py`.

This model class must be a child of the abstract `Network` class in `clinicadl/utils/network/network.py`.

Three abstract methods must be implemented to make it work
1. `forward`: computes the forward pass of the network, it may return several outputs
   needed to compute the loss.
2. `predict`: computes the forward pass of the network and only returns the main output.
3. `compute_outputs_and_loss`: computes the main outputs and the loss of the network.

If you want to implement a network which outputs an array, you can inherit from `CNN` class in
`clinicadl/utils/network/sub_network.py` (see for example `Conv5_FC3` `clinicadl/utils/network/cnn/models.py`
which is a child class of `CNN`).

If you want to implement a reconstruction autoencoder, you can inherit from `Autoencoder` class in
`clinicadl/utils/network/sub_network.py` (see for example `AE_Conv5_FC3` `clinicadl/utils/network/cnn/models.py`
which is a child class of `Autoencoder`).

Your network may be parametrized: in this case parameter names must correspond to the options of the
command line (for example `dropout`) or `input_size` / `output_size` which are computed by the MAPSManager. 
If you need a new parameter for your class you will have to add it to the command line.


## Custom input type

Input types that are already provided in `clinicadl` are `image`, `patch`, `roi` and `slice`. To add a custom input type, 
please follow the steps detailed below:

- Choose a mode name for this input type (for example default ones are image, patch, roi and slice). 
- Add your dataset class in `clinicadl/utils/caps_dataset/data.py` as a child class of the abstract class `CapsDataset`.
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

## Custom task

Available tasks in ClinicaDL are `classification`, `regression` and `reconstruction`.
You can implement a new task `task` by adding its corresponding TaskManager in 
`clinicadl/utils/task_manager/<task>.py`. This new class must be a child from the abstract class
`TaskManager` in `clinicadl/utils/task_manager/task_manager.py`.

Then modify the `_init_task_manager` function in the class `MapsManager` at 
`clinicadl/utils/maps_manager/maps_manager.py`.

## Custom metric

Other metrics can be added to the ones already available in ClinicaDL.
If the metric to add is composed of several words, please use the acronym
to name it (for example `balanced accuracy` becomes `BA`, `mean squared error`
becomes `MSE`).

Then add the method `<metric>_fn` to `MetricModule` in `clinicadl/utils/metric_module.py`,
where `metric` is the name of your metric in lower case (for example `balanced accuracy` function
is `ba_fn`).

This metric will only be used to evaluate specific tasks, then the `evaluation_metrics` property of 
the corresponding `TaskManager` must be updated in `clinicadl/utils/task_manager`.

Finally, to use this metric as a selection metric, please the `metric_optimum` dict in
`clinicadl/utils/metric_module.py`. The key is the name of your metric, and the content is respectively
`min` or `max` if the performance improves when the metric respectively decreases or increases.