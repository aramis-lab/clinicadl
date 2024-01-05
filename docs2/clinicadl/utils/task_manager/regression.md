# Regression

[Clinicadl Index](../../../README.md#clinicadl-index) /
[Clinicadl](../../index.md#clinicadl) /
[Utils](../index.md#utils) /
[Task Manager](./index.md#task-manager) /
Regression

> Auto-generated documentation for [clinicadl.utils.task_manager.regression](../../../../clinicadl/utils/task_manager/regression.py) module.

- [Regression](#regression)
  - [RegressionManager](#regressionmanager)
    - [RegressionManager().columns](#regressionmanager()columns)
    - [RegressionManager().compute_metrics](#regressionmanager()compute_metrics)
    - [RegressionManager().ensemble_prediction](#regressionmanager()ensemble_prediction)
    - [RegressionManager().evaluation_metrics](#regressionmanager()evaluation_metrics)
    - [RegressionManager.generate_label_code](#regressionmanagergenerate_label_code)
    - [RegressionManager.generate_sampler](#regressionmanagergenerate_sampler)
    - [RegressionManager().generate_test_row](#regressionmanager()generate_test_row)
    - [RegressionManager.get_criterion](#regressionmanagerget_criterion)
    - [RegressionManager.get_default_network](#regressionmanagerget_default_network)
    - [RegressionManager.output_size](#regressionmanageroutput_size)
    - [RegressionManager().save_outputs](#regressionmanager()save_outputs)

## RegressionManager

[Show source in regression.py:10](../../../../clinicadl/utils/task_manager/regression.py#L10)

#### Signature

```python
class RegressionManager(TaskManager):
    def __init__(self, mode):
        ...
```

#### See also

- [TaskManager](./task_manager.md#taskmanager)

### RegressionManager().columns

[Show source in regression.py:17](../../../../clinicadl/utils/task_manager/regression.py#L17)

#### Signature

```python
@property
def columns(self):
    ...
```

### RegressionManager().compute_metrics

[Show source in regression.py:46](../../../../clinicadl/utils/task_manager/regression.py#L46)

#### Signature

```python
def compute_metrics(self, results_df):
    ...
```

### RegressionManager().ensemble_prediction

[Show source in regression.py:90](../../../../clinicadl/utils/task_manager/regression.py#L90)

Compute the results at the image-level by assembling the results on parts of the image.

#### Arguments

- `performance_df` *pd.DataFrame* - results that need to be assembled.
- `validation_df` *pd.DataFrame* - results on the validation set used to compute the performance
    of each separate part of the image.
- `selection_threshold` *float* - with soft-voting method, allows to exclude some parts of the image
    if their associated performance is too low.
- `use_labels` *bool* - If True, metrics are computed and the label column values must be different
    from None.
- `method` *str* - method to assemble the results. Current implementation proposes only hard-voting.

#### Returns

df_final (pd.DataFrame) the results on the image level
results (Dict[str, float]) the metrics on the image level

#### Signature

```python
def ensemble_prediction(
    self,
    performance_df,
    validation_df,
    selection_threshold=None,
    use_labels=True,
    method="hard",
):
    ...
```

### RegressionManager().evaluation_metrics

[Show source in regression.py:27](../../../../clinicadl/utils/task_manager/regression.py#L27)

#### Signature

```python
@property
def evaluation_metrics(self):
    ...
```

### RegressionManager.generate_label_code

[Show source in regression.py:52](../../../../clinicadl/utils/task_manager/regression.py#L52)

#### Signature

```python
@staticmethod
def generate_label_code(df, label):
    ...
```

### RegressionManager.generate_sampler

[Show source in regression.py:60](../../../../clinicadl/utils/task_manager/regression.py#L60)

#### Signature

```python
@staticmethod
def generate_sampler(dataset, sampler_option="random", n_bins=5):
    ...
```

### RegressionManager().generate_test_row

[Show source in regression.py:35](../../../../clinicadl/utils/task_manager/regression.py#L35)

#### Signature

```python
def generate_test_row(self, idx, data, outputs):
    ...
```

### RegressionManager.get_criterion

[Show source in regression.py:150](../../../../clinicadl/utils/task_manager/regression.py#L150)

#### Signature

```python
@staticmethod
def get_criterion(criterion=None):
    ...
```

### RegressionManager.get_default_network

[Show source in regression.py:168](../../../../clinicadl/utils/task_manager/regression.py#L168)

#### Signature

```python
@staticmethod
def get_default_network():
    ...
```

### RegressionManager.output_size

[Show source in regression.py:56](../../../../clinicadl/utils/task_manager/regression.py#L56)

#### Signature

```python
@staticmethod
def output_size(input_size, df, label):
    ...
```

### RegressionManager().save_outputs

[Show source in regression.py:31](../../../../clinicadl/utils/task_manager/regression.py#L31)

#### Signature

```python
@property
def save_outputs(self):
    ...
```