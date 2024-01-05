# Iotools

[Clinicadl Index](../../../README.md#clinicadl-index) /
[Clinicadl](../../index.md#clinicadl) /
[Utils](../index.md#utils) /
[Maps Manager](./index.md#maps-manager) /
Iotools

> Auto-generated documentation for [clinicadl.utils.maps_manager.iotools](../../../../clinicadl/utils/maps_manager/iotools.py) module.

- [Iotools](#iotools)
  - [check_and_clean](#check_and_clean)
  - [check_and_complete](#check_and_complete)
  - [commandline_to_json](#commandline_to_json)
  - [cpuStats](#cpustats)
  - [memReport](#memreport)
  - [read_json](#read_json)
  - [write_requirements_version](#write_requirements_version)

## check_and_clean

[Show source in iotools.py:29](../../../../clinicadl/utils/maps_manager/iotools.py#L29)

#### Signature

```python
def check_and_clean(directory_path: Path):
    ...
```



## check_and_complete

[Show source in iotools.py:212](../../../../clinicadl/utils/maps_manager/iotools.py#L212)

This function initializes missing fields with missing values.
Some fields are mandatory and cannot be initialized by default; this will raise an issue if they are missing.

#### Arguments

- `options` - (dict) the options used for training.
- `random_search` - (bool) If True the options are looking for mandatory values of random-search.

#### Signature

```python
def check_and_complete(options, random_search=False):
    ...
```



## commandline_to_json

[Show source in iotools.py:37](../../../../clinicadl/utils/maps_manager/iotools.py#L37)

This is a function to write the python argparse object into a json file.
This helps for DL when searching for hyperparameters

#### Arguments

- `commandline` - (dict) dictionary with all the command line options values.
- `logger` - (logging object) writer to stdout and stderr.
- `filename` - (str) name of the JSON file.

:return:

#### Signature

```python
def commandline_to_json(commandline, logger=None, filename="commandline.json"):
    ...
```



## cpuStats

[Show source in iotools.py:350](../../../../clinicadl/utils/maps_manager/iotools.py#L350)

#### Signature

```python
def cpuStats():
    ...
```



## memReport

[Show source in iotools.py:335](../../../../clinicadl/utils/maps_manager/iotools.py#L335)

#### Signature

```python
def memReport():
    ...
```



## read_json

[Show source in iotools.py:80](../../../../clinicadl/utils/maps_manager/iotools.py#L80)

Read a json file to update options dictionary.
Ensures retro-compatibility with previous namings in clinicadl.

#### Arguments

- `options` - (dict) options of the model.
- `json_path` - (str) If given path to the json file, else found with options.model_path.
- `test` - (bool) If given the reader will ignore some options specific to data.
- `read_computational` - (bool) if set to True, the computational arguments are also read.

#### Returns

options (dict) options of the model updated

#### Signature

```python
def read_json(
    options=None, json_path: Path = None, test=False, read_computational=False
):
    ...
```



## write_requirements_version

[Show source in iotools.py:13](../../../../clinicadl/utils/maps_manager/iotools.py#L13)

#### Signature

```python
def write_requirements_version(output_path: Path):
    ...
```