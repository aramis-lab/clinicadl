# EarlyStopping

[Clinicadl Index](../../README.md#clinicadl-index) /
[Clinicadl](../index.md#clinicadl) /
[Utils](./index.md#utils) /
EarlyStopping

> Auto-generated documentation for [clinicadl.utils.early_stopping](../../../clinicadl/utils/early_stopping.py) module.

- [EarlyStopping](#earlystopping)
  - [EarlyStopping](#earlystopping-1)
    - [EarlyStopping().step](#earlystopping()step)

## EarlyStopping

[Show source in early_stopping.py:1](../../../clinicadl/utils/early_stopping.py#L1)

#### Signature

```python
class EarlyStopping(object):
    def __init__(self, mode="min", min_delta=0, patience=10):
        ...
```

### EarlyStopping().step

[Show source in early_stopping.py:15](../../../clinicadl/utils/early_stopping.py#L15)

#### Signature

```python
def step(self, metrics):
    ...
```