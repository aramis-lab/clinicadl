# Reconstruction

[Clinicadl Index](../../../README.md#clinicadl-index) /
[Clinicadl](../../index.md#clinicadl) /
[Utils](../index.md#utils) /
[Task Manager](./index.md#task-manager) /
Reconstruction

> Auto-generated documentation for [clinicadl.utils.task_manager.reconstruction](../../../../clinicadl/utils/task_manager/reconstruction.py) module.

- [Reconstruction](#reconstruction)
  - [ReconstructionManager](#reconstructionmanager)
    - [ReconstructionManager().columns](#reconstructionmanager()columns)
    - [ReconstructionManager().compute_metrics](#reconstructionmanager()compute_metrics)
    - [ReconstructionManager().ensemble_prediction](#reconstructionmanager()ensemble_prediction)
    - [ReconstructionManager().evaluation_metrics](#reconstructionmanager()evaluation_metrics)
    - [ReconstructionManager.generate_label_code](#reconstructionmanagergenerate_label_code)
    - [ReconstructionManager.generate_sampler](#reconstructionmanagergenerate_sampler)
    - [ReconstructionManager().generate_test_row](#reconstructionmanager()generate_test_row)
    - [ReconstructionManager.get_criterion](#reconstructionmanagerget_criterion)
    - [ReconstructionManager.get_default_network](#reconstructionmanagerget_default_network)
    - [ReconstructionManager.output_size](#reconstructionmanageroutput_size)
    - [ReconstructionManager().save_outputs](#reconstructionmanager()save_outputs)

## ReconstructionManager

[Show source in reconstruction.py:8](../../../../clinicadl/utils/task_manager/reconstruction.py#L8)

#### Signature

```python
class ReconstructionManager(TaskManager):
    def __init__(self, mode):
        ...
```

#### See also

- [TaskManager](./task_manager.md#taskmanager)

### ReconstructionManager().columns

[Show source in reconstruction.py:15](../../../../clinicadl/utils/task_manager/reconstruction.py#L15)

#### Signature

```python
@property
def columns(self):
    ...
```

### ReconstructionManager().compute_metrics

[Show source in reconstruction.py:43](../../../../clinicadl/utils/task_manager/reconstruction.py#L43)

#### Signature

```python
def compute_metrics(self, results_df):
    ...
```

### ReconstructionManager().ensemble_prediction

[Show source in reconstruction.py:70](../../../../clinicadl/utils/task_manager/reconstruction.py#L70)

Do not perform any ensemble prediction as it is not possible for reconstruction.

#### Arguments

- `performance_df` *pd.DataFrame* - results that need to be assembled.
- `validation_df` *pd.DataFrame* - results on the validation set used to compute the performance
    of each separate part of the image.
- `selection_threshold` *float* - with soft-voting method, allows to exclude some parts of the image
    if their associated performance is too low.
- `use_labels` *bool* - If True, metrics are computed and the label column values must be different
    from None.
- `method` *str* - method to assemble the results. Current implementation proposes soft or hard-voting.

#### Returns

None

#### Signature

```python
def ensemble_prediction(
    self,
    performance_df,
    validation_df,
    selection_threshold=None,
    use_labels=True,
    method="soft",
):
    ...
```

### ReconstructionManager().evaluation_metrics

[Show source in reconstruction.py:22](../../../../clinicadl/utils/task_manager/reconstruction.py#L22)

#### Signature

```python
@property
def evaluation_metrics(self):
    ...
```

### ReconstructionManager.generate_label_code

[Show source in reconstruction.py:53](../../../../clinicadl/utils/task_manager/reconstruction.py#L53)

#### Signature

```python
@staticmethod
def generate_label_code(df, label):
    ...
```

### ReconstructionManager.generate_sampler

[Show source in reconstruction.py:57](../../../../clinicadl/utils/task_manager/reconstruction.py#L57)

#### Signature

```python
@staticmethod
def generate_sampler(dataset, sampler_option="random", n_bins=5):
    ...
```

### ReconstructionManager().generate_test_row

[Show source in reconstruction.py:30](../../../../clinicadl/utils/task_manager/reconstruction.py#L30)

#### Signature

```python
def generate_test_row(self, idx, data, outputs):
    ...
```

### ReconstructionManager.get_criterion

[Show source in reconstruction.py:96](../../../../clinicadl/utils/task_manager/reconstruction.py#L96)

#### Signature

```python
@staticmethod
def get_criterion(criterion=None):
    ...
```

### ReconstructionManager.get_default_network

[Show source in reconstruction.py:114](../../../../clinicadl/utils/task_manager/reconstruction.py#L114)

#### Signature

```python
@staticmethod
def get_default_network():
    ...
```

### ReconstructionManager.output_size

[Show source in reconstruction.py:49](../../../../clinicadl/utils/task_manager/reconstruction.py#L49)

#### Signature

```python
@staticmethod
def output_size(input_size, df, label):
    ...
```

### ReconstructionManager().save_outputs

[Show source in reconstruction.py:26](../../../../clinicadl/utils/task_manager/reconstruction.py#L26)

#### Signature

```python
@property
def save_outputs(self):
    ...
```