# LogWriter

[Clinicadl Index](../../../README.md#clinicadl-index) /
[Clinicadl](../../index.md#clinicadl) /
[Utils](../index.md#utils) /
[Maps Manager](./index.md#maps-manager) /
LogWriter

> Auto-generated documentation for [clinicadl.utils.maps_manager.logwriter](../../../../clinicadl/utils/maps_manager/logwriter.py) module.

- [LogWriter](#logwriter)
  - [LogWriter](#logwriter-1)
    - [LogWriter().step](#logwriter()step)

## LogWriter

[Show source in logwriter.py:7](../../../../clinicadl/utils/maps_manager/logwriter.py#L7)

Write training logs in the MAPS

#### Signature

```python
class LogWriter:
    def __init__(
        self,
        maps_path: Path,
        evaluation_metrics,
        split,
        resume=False,
        beginning_epoch=0,
        network=None,
    ):
        ...
```

### LogWriter().step

[Show source in logwriter.py:67](../../../../clinicadl/utils/maps_manager/logwriter.py#L67)

Write a new row on the output file training.tsv.

#### Arguments

- `epoch` *int* - current epoch number
- `i` *int* - current iteration number
- `metrics_train` *Dict[str:float]* - metrics on the training set
- `metrics_valid` *Dict[str:float]* - metrics on the validation set
- `len_epoch` *int* - number of iterations in an epoch

#### Signature

```python
def step(self, epoch, i, metrics_train, metrics_valid, len_epoch):
    ...
```