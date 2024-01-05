# MapsManager

[Clinicadl Index](../../../README.md#clinicadl-index) /
[Clinicadl](../../index.md#clinicadl) /
[Utils](../index.md#utils) /
[Maps Manager](./index.md#maps-manager) /
MapsManager

> Auto-generated documentation for [clinicadl.utils.maps_manager.maps_manager](../../../../clinicadl/utils/maps_manager/maps_manager.py) module.

- [MapsManager](#mapsmanager)
  - [MapsManager](#mapsmanager-1)
    - [MapsManager().__getattr__](#mapsmanager()__getattr__)
    - [MapsManager().get_best_epoch](#mapsmanager()get_best_epoch)
    - [MapsManager().get_group_info](#mapsmanager()get_group_info)
    - [MapsManager().get_interpretation](#mapsmanager()get_interpretation)
    - [MapsManager().get_metrics](#mapsmanager()get_metrics)
    - [MapsManager().get_model](#mapsmanager()get_model)
    - [MapsManager().get_parameters](#mapsmanager()get_parameters)
    - [MapsManager().get_prediction](#mapsmanager()get_prediction)
    - [MapsManager().get_state_dict](#mapsmanager()get_state_dict)
    - [MapsManager().interpret](#mapsmanager()interpret)
    - [MapsManager().predict](#mapsmanager()predict)
    - [MapsManager().resume](#mapsmanager()resume)
    - [MapsManager().train](#mapsmanager()train)
    - [MapsManager.write_description_log](#mapsmanagerwrite_description_log)
    - [MapsManager.write_parameters](#mapsmanagerwrite_parameters)

## MapsManager

[Show source in maps_manager.py:46](../../../../clinicadl/utils/maps_manager/maps_manager.py#L46)

#### Signature

```python
class MapsManager:
    def __init__(
        self, maps_path: Path, parameters: Dict[str, Any] = None, verbose: str = "info"
    ):
        ...
```

### MapsManager().__getattr__

[Show source in maps_manager.py:111](../../../../clinicadl/utils/maps_manager/maps_manager.py#L111)

Allow to directly get the values in parameters attribute

#### Signature

```python
def __getattr__(self, name):
    ...
```

### MapsManager().get_best_epoch

[Show source in maps_manager.py:2045](../../../../clinicadl/utils/maps_manager/maps_manager.py#L2045)

#### Signature

```python
def get_best_epoch(
    self, split: int = 0, selection_metric: str = None, network: int = None
) -> int:
    ...
```

### MapsManager().get_group_info

[Show source in maps_manager.py:1993](../../../../clinicadl/utils/maps_manager/maps_manager.py#L1993)

Gets information from corresponding data group
(list of participant_id / session_id + configuration parameters).
split is only needed if data_group is train or validation.

#### Signature

```python
def get_group_info(
    self, data_group: str, split: int = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    ...
```

### MapsManager().get_interpretation

[Show source in maps_manager.py:2172](../../../../clinicadl/utils/maps_manager/maps_manager.py#L2172)

Get the individual interpretation maps for one session if participant_id and session_id are filled.
Else load the mean interpretation map.

#### Arguments

- `data_group` *str* - Name of the data group used for the interpretation task.
- `name` *str* - name of the interpretation task.
- `split` *int* - Index of the split used for training.
- `selection_metric` *str* - Metric used for best weights selection.
- `verbose` *bool* - if True will print associated prediction.log.
- `participant_id` *str* - ID of the participant (if not given load mean map).
- `session_id` *str* - ID of the session (if not give load the mean map).
- `mode_id` *int* - Index of the mode used.

#### Returns

- `(torch.Tensor)` - Tensor of the interpretability map.

#### Signature

```python
def get_interpretation(
    self,
    data_group,
    name,
    split=0,
    selection_metric=None,
    verbose=True,
    participant_id=None,
    session_id=None,
    mode_id=0,
) -> torch.Tensor:
    ...
```

### MapsManager().get_metrics

[Show source in maps_manager.py:2139](../../../../clinicadl/utils/maps_manager/maps_manager.py#L2139)

Get the metrics corresponding to a group of participants identified by its data_group.

#### Arguments

- `data_group` *str* - name of the data group used for the prediction task.
- `split` *int* - Index of the split used for training.
- `selection_metric` *str* - Metric used for best weights selection.
- `mode` *str* - level of the prediction
- `verbose` *bool* - if True will print associated prediction.log

#### Returns

- `(dict[str:float])` - Values of the metrics

#### Signature

```python
def get_metrics(
    self, data_group, split=0, selection_metric=None, mode="image", verbose=True
):
    ...
```

### MapsManager().get_model

[Show source in maps_manager.py:2032](../../../../clinicadl/utils/maps_manager/maps_manager.py#L2032)

#### Signature

```python
def get_model(
    self, split: int = 0, selection_metric: str = None, network: int = None
) -> Network:
    ...
```

#### See also

- [Network](../network/network.md#network)

### MapsManager().get_parameters

[Show source in maps_manager.py:2027](../../../../clinicadl/utils/maps_manager/maps_manager.py#L2027)

Returns the training parameters dictionary.

#### Signature

```python
def get_parameters(self):
    ...
```

### MapsManager().get_prediction

[Show source in maps_manager.py:2102](../../../../clinicadl/utils/maps_manager/maps_manager.py#L2102)

Get the individual predictions for each participant corresponding to one group
of participants identified by its data group.

#### Arguments

- `data_group` *str* - name of the data group used for the prediction task.
- `split` *int* - Index of the split used for training.
- `selection_metric` *str* - Metric used for best weights selection.
- `mode` *str* - level of the prediction.
- `verbose` *bool* - if True will print associated prediction.log.

#### Returns

- `(DataFrame)` - Results indexed by columns 'participant_id' and 'session_id' which
identifies the image in the BIDS / CAPS.

#### Signature

```python
def get_prediction(
    self, data_group, split=0, selection_metric=None, mode="image", verbose=False
):
    ...
```

### MapsManager().get_state_dict

[Show source in maps_manager.py:2058](../../../../clinicadl/utils/maps_manager/maps_manager.py#L2058)

Get the model trained corresponding to one split and one metric evaluated on the validation set.

#### Arguments

- `split` *int* - Index of the split used for training.
- `selection_metric` *str* - name of the metric used for the selection.
- `network` *int* - Index of the network trained (used in multi-network setting only).
- `map_location` *str* - torch.device object or a string containing a device tag,
    it indicates the location where all tensors should be loaded.
    (see https://pytorch.org/docs/stable/generated/torch.load.html).

#### Returns

- `(Dict)` - dictionary of results (weights, epoch number, metrics values)

#### Signature

```python
def get_state_dict(
    self, split=0, selection_metric=None, network=None, map_location=None
):
    ...
```

### MapsManager().interpret

[Show source in maps_manager.py:375](../../../../clinicadl/utils/maps_manager/maps_manager.py#L375)

Performs the interpretation task on a subset of caps_directory defined in a TSV file.
The mean interpretation is always saved, to save the individual interpretations set save_individual to True.

Parameters
----------
data_group: str
    Name of the data group interpreted.
name: str
    Name of the interpretation procedure.
method: str
    Method used for extraction (ex: gradients, grad-cam...).
caps_directory: str (Path)
    Path to the CAPS folder. For more information please refer to
    [clinica documentation](https://aramislab.paris.inria.fr/clinica/docs/public/latest/CAPS/Introduction/).
    Default will load the value of an existing data group.
tsv_path: str (Path)
    Path to a TSV file containing the list of participants and sessions to test.
    Default will load the DataFrame of an existing data group.
split_list: list of int
    List of splits to interpret. Default perform interpretation on all splits available.
selection_metrics: list of str
    List of selection metrics to interpret.
    Default performs the interpretation on all selection metrics available.
multi_cohort: bool
    If True considers that tsv_path is the path to a multi-cohort TSV.
diagnoses: list of str
    List of diagnoses to load if tsv_path is a split_directory.
    Default uses the same as in training step.
target_node: int
    Node from which the interpretation is computed.
save_individual: bool
    If True saves the individual map of each participant / session couple.
batch_size: int
    If given, sets the value of batch_size, else use the same as in training step.
n_proc: int
    If given, sets the value of num_workers, else use the same as in training step.
gpu: bool
    If given, a new value for the device of the model will be computed.
overwrite: bool
    If True erase the occurrences of data_group.
overwrite_name: bool
    If True erase the occurrences of name.
level: int
    Layer number in the convolutional part after which the feature map is chosen.
save_nifi : bool
    If True, save the interpretation map in nifti format.

#### Signature

```python
def interpret(
    self,
    data_group,
    name,
    method,
    caps_directory: Path = None,
    tsv_path: Path = None,
    split_list=None,
    selection_metrics=None,
    multi_cohort=False,
    diagnoses=(),
    target_node=0,
    save_individual=False,
    batch_size=None,
    n_proc=None,
    gpu=None,
    overwrite=False,
    overwrite_name=False,
    level=None,
    save_nifti=False,
):
    ...
```

### MapsManager().predict

[Show source in maps_manager.py:182](../../../../clinicadl/utils/maps_manager/maps_manager.py#L182)

Performs the prediction task on a subset of caps_directory defined in a TSV file.

#### Arguments

- `data_group` - name of the data group tested.
- `caps_directory` - path to the CAPS folder. For more information please refer to
    [clinica documentation](https://aramislab.paris.inria.fr/clinica/docs/public/latest/CAPS/Introduction/).
    Default will load the value of an existing data group
- `tsv_path` - path to a TSV file containing the list of participants and sessions to test.
    Default will load the DataFrame of an existing data group
- `split_list` - list of splits to test. Default perform prediction on all splits available.
- `selection_metrics` *list[str]* - list of selection metrics to test.
    Default performs the prediction on all selection metrics available.
- `multi_cohort` - If True considers that tsv_path is the path to a multi-cohort TSV.
- `diagnoses` - List of diagnoses to load if tsv_path is a split_directory.
    Default uses the same as in training step.
- `use_labels` - If True, the labels must exist in test meta-data and metrics are computed.
- `batch_size` - If given, sets the value of batch_size, else use the same as in training step.
- `n_proc` - If given, sets the value of num_workers, else use the same as in training step.
- `gpu` - If given, a new value for the device of the model will be computed.
- `overwrite` - If True erase the occurrences of data_group.
- `label` - Target label used for training (if network_task in [`regression`, `classification`]).
- `label_code` - dictionary linking the target values to a node number.

#### Signature

```python
def predict(
    self,
    data_group: str,
    caps_directory: Path = None,
    tsv_path: Path = None,
    split_list: List[int] = None,
    selection_metrics: List[str] = None,
    multi_cohort: bool = False,
    diagnoses: List[str] = (),
    use_labels: bool = True,
    batch_size: int = None,
    n_proc: int = None,
    gpu: bool = None,
    overwrite: bool = False,
    label: str = None,
    label_code: Optional[Dict[str, int]] = "default",
    save_tensor: bool = False,
    save_nifti: bool = False,
):
    ...
```

### MapsManager().resume

[Show source in maps_manager.py:153](../../../../clinicadl/utils/maps_manager/maps_manager.py#L153)

Resumes the training task for a defined list of splits.

#### Arguments

- `split_list` - list of splits on which the training task is performed.
    Default trains all splits.

#### Raises

- `MAPSError` - If splits specified in input do not exist.

#### Signature

```python
def resume(self, split_list: List[int] = None):
    ...
```

### MapsManager().train

[Show source in maps_manager.py:118](../../../../clinicadl/utils/maps_manager/maps_manager.py#L118)

Performs the training task for a defined list of splits

#### Arguments

- `split_list` - list of splits on which the training task is performed.
    Default trains all splits of the cross-validation.
- `overwrite` - If True previously trained splits that are going to be trained
    are erased.

#### Raises

- `MAPSError` - If splits specified in input already exist and overwrite is False.

#### Signature

```python
def train(self, split_list: List[int] = None, overwrite: bool = False):
    ...
```

### MapsManager.write_description_log

[Show source in maps_manager.py:1630](../../../../clinicadl/utils/maps_manager/maps_manager.py#L1630)

Write description log file associated to a data group.

#### Arguments

- `log_dir` *str* - path to the log file directory.
- `data_group` *str* - name of the data group used for the task.
caps_dict (dict[str, str]): Dictionary of the CAPS folders used for the task
- `df` *pd.DataFrame* - DataFrame of the meta-data used for the task.

#### Signature

```python
@staticmethod
def write_description_log(log_dir: Path, data_group, caps_dict, df):
    ...
```

### MapsManager.write_parameters

[Show source in maps_manager.py:1429](../../../../clinicadl/utils/maps_manager/maps_manager.py#L1429)

#### Signature

```python
@staticmethod
def write_parameters(json_path: Path, parameters, verbose=True):
    ...
```