# TaskManager

[Clinicadl Index](../../../README.md#clinicadl-index) /
[Clinicadl](../../index.md#clinicadl) /
[Utils](../index.md#utils) /
[Task Manager](./index.md#task-manager) /
TaskManager

> Auto-generated documentation for [clinicadl.utils.task_manager.task_manager](../../../../clinicadl/utils/task_manager/task_manager.py) module.

- [TaskManager](#taskmanager)
  - [TaskManager](#taskmanager-1)
    - [TaskManager().columns](#taskmanager()columns)
    - [TaskManager().compute_metrics](#taskmanager()compute_metrics)
    - [TaskManager().ensemble_prediction](#taskmanager()ensemble_prediction)
    - [TaskManager().evaluation_metrics](#taskmanager()evaluation_metrics)
    - [TaskManager.generate_label_code](#taskmanagergenerate_label_code)
    - [TaskManager.generate_sampler](#taskmanagergenerate_sampler)
    - [TaskManager().generate_test_row](#taskmanager()generate_test_row)
    - [TaskManager.get_criterion](#taskmanagerget_criterion)
    - [TaskManager.get_default_network](#taskmanagerget_default_network)
    - [TaskManager.output_size](#taskmanageroutput_size)
    - [TaskManager().save_outputs](#taskmanager()save_outputs)
    - [TaskManager().test](#taskmanager()test)

## TaskManager

[Show source in task_manager.py:17](../../../../clinicadl/utils/task_manager/task_manager.py#L17)

#### Signature

```python
class TaskManager:
    def __init__(self, mode: str, n_classes: int = None):
        ...
```

### TaskManager().columns

[Show source in task_manager.py:22](../../../../clinicadl/utils/task_manager/task_manager.py#L22)

List of the columns' names in the TSV file containing the predictions.

#### Signature

```python
@property
@abstractmethod
def columns(self):
    ...
```

### TaskManager().compute_metrics

[Show source in task_manager.py:62](../../../../clinicadl/utils/task_manager/task_manager.py#L62)

Compute the metrics based on the result of generate_test_row

#### Arguments

- `results_df` - results generated based on _results_test_row

#### Returns

dictionary of metrics

#### Signature

```python
@abstractmethod
def compute_metrics(self, results_df: pd.DataFrame) -> Dict[str, float]:
    ...
```

### TaskManager().ensemble_prediction

[Show source in task_manager.py:74](../../../../clinicadl/utils/task_manager/task_manager.py#L74)

Compute the results at the image-level by assembling the results on parts of the image.

#### Arguments

- `performance_df` - results that need to be assembled.
- `validation_df` - results on the validation set used to compute the performance
    of each separate part of the image.
- `selection_threshold` - with soft-voting method, allows to exclude some parts of the image
    if their associated performance is too low.
- `use_labels` - If True, metrics are computed and the label column values must be different
    from None.
- `method` - method to assemble the results. Current implementation proposes soft or hard-voting.

#### Returns

the results and metrics on the image level

#### Signature

```python
@abstractmethod
def ensemble_prediction(
    self,
    performance_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    selection_threshold: float = None,
    use_labels: bool = True,
    method: str = "soft",
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    ...
```

### TaskManager().evaluation_metrics

[Show source in task_manager.py:30](../../../../clinicadl/utils/task_manager/task_manager.py#L30)

Evaluation metrics which can be used to evaluate the task.

#### Signature

```python
@property
@abstractmethod
def evaluation_metrics(self):
    ...
```

### TaskManager.generate_label_code

[Show source in task_manager.py:101](../../../../clinicadl/utils/task_manager/task_manager.py#L101)

Generates a label code that links the output node number to label value.

#### Arguments

- `df` - meta-data of the training set.
- `label` - name of the column containing the labels.

#### Returns

label_code

#### Signature

```python
@staticmethod
@abstractmethod
def generate_label_code(df: pd.DataFrame, label: str) -> Optional[Dict[str, int]]:
    ...
```

### TaskManager.generate_sampler

[Show source in task_manager.py:132](../../../../clinicadl/utils/task_manager/task_manager.py#L132)

Returns sampler according to the wanted options.

#### Arguments

- `dataset` - the dataset to sample from.
- `sampler_option` - choice of sampler.
- `n_bins` - number of bins to used for a continuous variable (regression task).

#### Returns

callable given to the training data loader.

#### Signature

```python
@staticmethod
@abstractmethod
def generate_sampler(
    dataset: CapsDataset, sampler_option: str = "random", n_bins: int = 5
) -> Sampler:
    ...
```

#### See also

- [CapsDataset](../caps_dataset/data.md#capsdataset)

### TaskManager().generate_test_row

[Show source in task_manager.py:46](../../../../clinicadl/utils/task_manager/task_manager.py#L46)

Computes an individual row of the prediction TSV file.

#### Arguments

- `idx` - index of the individual input and output in the batch.
- `data` - input batch generated by a DataLoader on a CapsDataset.
- `outputs` - output batch generated by a forward pass in the model.

#### Returns

list of items to be contained in a row of the prediction TSV file.

#### Signature

```python
@abstractmethod
def generate_test_row(
    self, idx: int, data: Dict[str, Any], outputs: Tensor
) -> List[List[Any]]:
    ...
```

### TaskManager.get_criterion

[Show source in task_manager.py:149](../../../../clinicadl/utils/task_manager/task_manager.py#L149)

Gives the optimization criterion.
Must check that it is compatible with the task.

#### Arguments

- `criterion` - name of the loss as written in Pytorch.

#### Raises

- `ClinicaDLArgumentError` - if the criterion is not compatible with the task.

#### Signature

```python
@staticmethod
@abstractmethod
def get_criterion(criterion: str = None) -> _Loss:
    ...
```

### TaskManager.get_default_network

[Show source in task_manager.py:163](../../../../clinicadl/utils/task_manager/task_manager.py#L163)

Returns the default network to use when no architecture is specified.

#### Signature

```python
@staticmethod
@abstractmethod
def get_default_network() -> Network:
    ...
```

#### See also

- [Network](../network/network.md#network)

### TaskManager.output_size

[Show source in task_manager.py:115](../../../../clinicadl/utils/task_manager/task_manager.py#L115)

Computes the output_size needed to perform the task.

#### Arguments

- `input_size` - size of the input.
- `df` - meta-data of the training set.
- `label` - name of the column containing the labels.

#### Returns

output_size

#### Signature

```python
@staticmethod
@abstractmethod
def output_size(
    input_size: Sequence[int], df: pd.DataFrame, label: str
) -> Sequence[int]:
    ...
```

### TaskManager().save_outputs

[Show source in task_manager.py:38](../../../../clinicadl/utils/task_manager/task_manager.py#L38)

Boolean value indicating if the output values should be saved as tensor for this task.

#### Signature

```python
@property
@abstractmethod
def save_outputs(self):
    ...
```

### TaskManager().test

[Show source in task_manager.py:169](../../../../clinicadl/utils/task_manager/task_manager.py#L169)

Computes the predictions and evaluation metrics.

#### Arguments

- `model` - the model trained.
- `dataloader` - wrapper of a CapsDataset.
- `criterion` - function to calculate the loss.
- `use_labels` - If True the true_label will be written in output DataFrame
    and metrics dict will be created.

#### Returns

the results and metrics on the image level.

#### Signature

```python
def test(
    self,
    model: Network,
    dataloader: DataLoader,
    criterion: _Loss,
    use_labels: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    ...
```

#### See also

- [Network](../network/network.md#network)