# Maps Manager Utils

[Clinicadl Index](../../../README.md#clinicadl-index) /
[Clinicadl](../../index.md#clinicadl) /
[Utils](../index.md#utils) /
[Maps Manager](./index.md#maps-manager) /
Maps Manager Utils

> Auto-generated documentation for [clinicadl.utils.maps_manager.maps_manager_utils](../../../../clinicadl/utils/maps_manager/maps_manager_utils.py) module.

- [Maps Manager Utils](#maps-manager-utils)
  - [add_default_values](#add_default_values)
  - [change_path_to_str](#change_path_to_str)
  - [change_str_to_path](#change_str_to_path)
  - [read_json](#read_json)
  - [remove_unused_tasks](#remove_unused_tasks)

## add_default_values

[Show source in maps_manager_utils.py:10](../../../../clinicadl/utils/maps_manager/maps_manager_utils.py#L10)

Updates the training parameters defined by the user with the default values in missing fields.

#### Arguments

- `user_dict` - dictionary of training parameters defined by the user.

#### Returns

dictionary of values ready to use for the training process.

#### Signature

```python
def add_default_values(user_dict: Dict[str, Any]) -> Dict[str, Any]:
    ...
```



## change_path_to_str

[Show source in maps_manager_utils.py:221](../../../../clinicadl/utils/maps_manager/maps_manager_utils.py#L221)

For all paths in the dictionnary, it changes the type from pathlib.Path to str.

Paramaters
----------
toml_dict: Dict[str, Dict[str, Any]]
    Dictionary of options as written in a TOML file, with type(path)=pathlib.Path

Returns
-------
    Updated TOML dictionary with type(path)=str

#### Signature

```python
def change_path_to_str(
    toml_dict: Dict[str, Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    ...
```



## change_str_to_path

[Show source in maps_manager_utils.py:175](../../../../clinicadl/utils/maps_manager/maps_manager_utils.py#L175)

For all paths in the dictionnary, it changes the type from str to pathlib.Path.

Paramaters
----------
toml_dict: Dict[str, Dict[str, Any]]
    Dictionary of options as written in a TOML file, with type(path)=str

Returns
-------
    Updated TOML dictionary with type(path)=pathlib.Path

#### Signature

```python
def change_str_to_path(
    toml_dict: Dict[str, Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    ...
```



## read_json

[Show source in maps_manager_utils.py:43](../../../../clinicadl/utils/maps_manager/maps_manager_utils.py#L43)

Ensures retro-compatibility between the different versions of ClinicaDL.

#### Arguments

- `json_path` - path to the JSON file summing the parameters of a MAPS.

#### Returns

dictionary of training parameters.

#### Signature

```python
def read_json(json_path: Path) -> Dict[str, Any]:
    ...
```



## remove_unused_tasks

[Show source in maps_manager_utils.py:145](../../../../clinicadl/utils/maps_manager/maps_manager_utils.py#L145)

Remove options depending on other tasks than task

#### Arguments

- `toml_dict` - dictionary of options as written in a TOML file.
- `task` - task which will be learnt by the network.

#### Returns

updated TOML dictionary.

#### Signature

```python
def remove_unused_tasks(
    toml_dict: Dict[str, Dict[str, Any]], task: str
) -> Dict[str, Dict[str, Any]]:
    ...
```