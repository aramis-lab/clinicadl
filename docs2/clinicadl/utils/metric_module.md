# MetricModule

[Clinicadl Index](../../README.md#clinicadl-index) /
[Clinicadl](../index.md#clinicadl) /
[Utils](./index.md#utils) /
MetricModule

> Auto-generated documentation for [clinicadl.utils.metric_module](../../../clinicadl/utils/metric_module.py) module.

- [MetricModule](#metricmodule)
  - [MetricModule](#metricmodule-1)
    - [MetricModule.accuracy_fn](#metricmoduleaccuracy_fn)
    - [MetricModule().apply](#metricmodule()apply)
    - [MetricModule.ba_fn](#metricmoduleba_fn)
    - [MetricModule.confusion_matrix_fn](#metricmoduleconfusion_matrix_fn)
    - [MetricModule.mae_fn](#metricmodulemae_fn)
    - [MetricModule.mse_fn](#metricmodulemse_fn)
    - [MetricModule.npv_fn](#metricmodulenpv_fn)
    - [MetricModule.ppv_fn](#metricmoduleppv_fn)
    - [MetricModule.psnr_fn](#metricmodulepsnr_fn)
    - [MetricModule.sensitivity_fn](#metricmodulesensitivity_fn)
    - [MetricModule.specificity_fn](#metricmodulespecificity_fn)
    - [MetricModule.ssim_fn](#metricmodulessim_fn)
  - [RetainBest](#retainbest)
    - [RetainBest().set_optimum](#retainbest()set_optimum)
    - [RetainBest().step](#retainbest()step)

## MetricModule

[Show source in metric_module.py:23](../../../clinicadl/utils/metric_module.py#L23)

#### Signature

```python
class MetricModule:
    def __init__(self, metrics, n_classes=2):
        ...
```

### MetricModule.accuracy_fn

[Show source in metric_module.py:99](../../../clinicadl/utils/metric_module.py#L99)

#### Arguments

- `y` *List* - list of labels
- `y_pred` *List* - list of predictions

#### Returns

(float) accuracy

#### Signature

```python
@staticmethod
def accuracy_fn(y, y_pred):
    ...
```

### MetricModule().apply

[Show source in metric_module.py:43](../../../clinicadl/utils/metric_module.py#L43)

This is a function to calculate the different metrics based on the list of true label and predicted label

#### Arguments

- `y` *List* - list of labels
- `y_pred` *List* - list of predictions

#### Returns

(Dict[str:float]) metrics results

#### Signature

```python
def apply(self, y, y_pred):
    ...
```

### MetricModule.ba_fn

[Show source in metric_module.py:184](../../../clinicadl/utils/metric_module.py#L184)

#### Arguments

- `y` *List* - list of labels
- `y_pred` *List* - list of predictions
- `class_number` *int* - number of the class studied

#### Returns

(float) balanced accuracy

#### Signature

```python
@staticmethod
def ba_fn(y, y_pred, class_number):
    ...
```

### MetricModule.confusion_matrix_fn

[Show source in metric_module.py:200](../../../clinicadl/utils/metric_module.py#L200)

#### Arguments

- `y` *List* - list of labels
- `y_pred` *List* - list of predictions

#### Returns

(Dict[str:float]) confusion matrix

#### Signature

```python
@staticmethod
def confusion_matrix_fn(y, y_pred):
    ...
```

### MetricModule.mae_fn

[Show source in metric_module.py:75](../../../clinicadl/utils/metric_module.py#L75)

#### Arguments

- `y` *List* - list of labels
- `y_pred` *List* - list of predictions

#### Returns

(float) mean absolute error

#### Signature

```python
@staticmethod
def mae_fn(y, y_pred):
    ...
```

### MetricModule.mse_fn

[Show source in metric_module.py:87](../../../clinicadl/utils/metric_module.py#L87)

#### Arguments

- `y` *List* - list of labels
- `y_pred` *List* - list of predictions

#### Returns

(float) mean squared error

#### Signature

```python
@staticmethod
def mse_fn(y, y_pred):
    ...
```

### MetricModule.npv_fn

[Show source in metric_module.py:166](../../../clinicadl/utils/metric_module.py#L166)

#### Arguments

- `y` *List* - list of labels
- `y_pred` *List* - list of predictions
- `class_number` *int* - number of the class studied

#### Returns

(float) negative predictive value

#### Signature

```python
@staticmethod
def npv_fn(y, y_pred, class_number):
    ...
```

### MetricModule.ppv_fn

[Show source in metric_module.py:148](../../../clinicadl/utils/metric_module.py#L148)

#### Arguments

- `y` *List* - list of labels
- `y_pred` *List* - list of predictions
- `class_number` *int* - number of the class studied

#### Returns

(float) positive predictive value

#### Signature

```python
@staticmethod
def ppv_fn(y, y_pred, class_number):
    ...
```

### MetricModule.psnr_fn

[Show source in metric_module.py:234](../../../clinicadl/utils/metric_module.py#L234)

#### Arguments

- `y` *List* - list of labels
- `y_pred` *List* - list of predictions

#### Returns

(float) PSNR

#### Signature

```python
@staticmethod
def psnr_fn(y, y_pred):
    ...
```

### MetricModule.sensitivity_fn

[Show source in metric_module.py:112](../../../clinicadl/utils/metric_module.py#L112)

#### Arguments

- `y` *List* - list of labels
- `y_pred` *List* - list of predictions
- `class_number` *int* - number of the class studied

#### Returns

(float) sensitivity

#### Signature

```python
@staticmethod
def sensitivity_fn(y, y_pred, class_number):
    ...
```

### MetricModule.specificity_fn

[Show source in metric_module.py:130](../../../clinicadl/utils/metric_module.py#L130)

#### Arguments

- `y` *List* - list of labels
- `y_pred` *List* - list of predictions
- `class_number` *int* - number of the class studied

#### Returns

(float) specificity

#### Signature

```python
@staticmethod
def specificity_fn(y, y_pred, class_number):
    ...
```

### MetricModule.ssim_fn

[Show source in metric_module.py:221](../../../clinicadl/utils/metric_module.py#L221)

#### Arguments

- `y` *List* - list of labels
- `y_pred` *List* - list of predictions

#### Returns

(float) SSIM

#### Signature

```python
@staticmethod
def ssim_fn(y, y_pred):
    ...
```



## RetainBest

[Show source in metric_module.py:248](../../../clinicadl/utils/metric_module.py#L248)

A class to retain the best and overfitting values for a set of wanted metrics.

#### Signature

```python
class RetainBest:
    def __init__(self, selection_metrics: List[str], n_classes: int = 0):
        ...
```

### RetainBest().set_optimum

[Show source in metric_module.py:282](../../../clinicadl/utils/metric_module.py#L282)

#### Signature

```python
def set_optimum(self, selection: str):
    ...
```

### RetainBest().step

[Show source in metric_module.py:293](../../../clinicadl/utils/metric_module.py#L293)

Computes for each metric if this is the best value ever seen.

#### Arguments

- `metrics_valid` - metrics computed on the validation set

#### Returns

metric is associated to True if it is the best value ever seen.

#### Signature

```python
def step(self, metrics_valid: Dict[str, float]) -> Dict[str, bool]:
    ...
```