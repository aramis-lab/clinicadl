# Train Utils

[Clinicadl Index](../../README.md#clinicadl-index) /
[Clinicadl](../index.md#clinicadl) /
[Train](./index.md#train) /
Train Utils

> Auto-generated documentation for [clinicadl.train.train_utils](../../../clinicadl/train/train_utils.py) module.

- [Train Utils](#train-utils)
  - [build_train_dict](#build_train_dict)
  - [get_model_list](#get_model_list)

## build_train_dict

[Show source in train_utils.py:14](../../../clinicadl/train/train_utils.py#L14)

Read the configuration file given by the user.
If it is a TOML file, ensures that the format corresponds to the one in resources.

#### Arguments

- `config_file` - path to a configuration file (JSON of TOML).
- `task` - task learnt by the network (example: classification, regression, reconstruction...).

#### Returns

dictionary of values ready to use for the MapsManager

#### Signature

```python
def build_train_dict(config_file: Path, task: str) -> Dict[str, Any]:
    ...
```



## get_model_list

[Show source in train_utils.py:88](../../../clinicadl/train/train_utils.py#L88)

Print the list of models available in ClinicaDL.
If --architecture is given, information about how to use this model will be displayed.
If --model_layers flag is added, this pipeline will show the whole model layers.
If --input_size is added, it will show the whole model layers with chosen input shape.

#### Signature

```python
def get_model_list(architecture=None, input_size=None, model_layers=False):
    ...
```