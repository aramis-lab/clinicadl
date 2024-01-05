# Classification

[Clinicadl Index](../../../README.md#clinicadl-index) /
[Clinicadl](../../index.md#clinicadl) /
[Utils](../index.md#utils) /
[Task Manager](./index.md#task-manager) /
Classification

> Auto-generated documentation for [clinicadl.utils.task_manager.classification](../../../../clinicadl/utils/task_manager/classification.py) module.

- [Classification](#classification)
  - [ClassificationManager](#classificationmanager)
    - [ClassificationManager().columns](#classificationmanager()columns)
    - [ClassificationManager().compute_metrics](#classificationmanager()compute_metrics)
    - [ClassificationManager().ensemble_prediction](#classificationmanager()ensemble_prediction)
    - [ClassificationManager().evaluation_metrics](#classificationmanager()evaluation_metrics)
    - [ClassificationManager.generate_label_code](#classificationmanagergenerate_label_code)
    - [ClassificationManager.generate_sampler](#classificationmanagergenerate_sampler)
    - [ClassificationManager().generate_test_row](#classificationmanager()generate_test_row)
    - [ClassificationManager.get_criterion](#classificationmanagerget_criterion)
    - [ClassificationManager.get_default_network](#classificationmanagerget_default_network)
    - [ClassificationManager.output_size](#classificationmanageroutput_size)
    - [ClassificationManager().save_outputs](#classificationmanager()save_outputs)

## ClassificationManager

[Show source in classification.py:17](../../../../clinicadl/utils/task_manager/classification.py#L17)

#### Signature

```python
class ClassificationManager(TaskManager):
    def __init__(self, mode, n_classes=None, df=None, label=None):
        ...
```

#### See also

- [TaskManager](./task_manager.md#taskmanager)

### ClassificationManager().columns

[Show source in classification.py:30](../../../../clinicadl/utils/task_manager/classification.py#L30)

#### Signature

```python
@property
def columns(self):
    ...
```

### ClassificationManager().compute_metrics

[Show source in classification.py:63](../../../../clinicadl/utils/task_manager/classification.py#L63)

#### Signature

```python
def compute_metrics(self, results_df):
    ...
```

### ClassificationManager().ensemble_prediction

[Show source in classification.py:110](../../../../clinicadl/utils/task_manager/classification.py#L110)

Computes hard or soft voting based on the probabilities in performance_df. Weights are computed based
on the balanced accuracies of validation_df.

ref: S. Raschka. Python Machine Learning., 2015

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
    method="soft",
):
    ...
```

### ClassificationManager().evaluation_metrics

[Show source in classification.py:40](../../../../clinicadl/utils/task_manager/classification.py#L40)

#### Signature

```python
@property
def evaluation_metrics(self):
    ...
```

### ClassificationManager.generate_label_code

[Show source in classification.py:69](../../../../clinicadl/utils/task_manager/classification.py#L69)

#### Signature

```python
@staticmethod
def generate_label_code(df, label):
    ...
```

### ClassificationManager.generate_sampler

[Show source in classification.py:80](../../../../clinicadl/utils/task_manager/classification.py#L80)

#### Signature

```python
@staticmethod
def generate_sampler(dataset, sampler_option="random", n_bins=5):
    ...
```

### ClassificationManager().generate_test_row

[Show source in classification.py:48](../../../../clinicadl/utils/task_manager/classification.py#L48)

#### Signature

```python
def generate_test_row(self, idx, data, outputs):
    ...
```

### ClassificationManager.get_criterion

[Show source in classification.py:193](../../../../clinicadl/utils/task_manager/classification.py#L193)

#### Signature

```python
@staticmethod
def get_criterion(criterion=None):
    ...
```

### ClassificationManager.get_default_network

[Show source in classification.py:204](../../../../clinicadl/utils/task_manager/classification.py#L204)

#### Signature

```python
@staticmethod
def get_default_network():
    ...
```

### ClassificationManager.output_size

[Show source in classification.py:75](../../../../clinicadl/utils/task_manager/classification.py#L75)

#### Signature

```python
@staticmethod
def output_size(input_size, df, label):
    ...
```

### ClassificationManager().save_outputs

[Show source in classification.py:44](../../../../clinicadl/utils/task_manager/classification.py#L44)

#### Signature

```python
@property
def save_outputs(self):
    ...
```