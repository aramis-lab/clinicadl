# Kfold

[Clinicadl Index](../../../README.md#clinicadl-index) /
[Clinicadl](../../index.md#clinicadl) /
[Utils](../index.md#utils) /
[Split Manager](./index.md#split-manager) /
Kfold

> Auto-generated documentation for [clinicadl.utils.split_manager.kfold](../../../../clinicadl/utils/split_manager/kfold.py) module.

- [Kfold](#kfold)
  - [KFoldSplit](#kfoldsplit)
    - [KFoldSplit().allowed_splits_list](#kfoldsplit()allowed_splits_list)
    - [KFoldSplit().max_length](#kfoldsplit()max_length)
    - [KFoldSplit().split_iterator](#kfoldsplit()split_iterator)

## KFoldSplit

[Show source in kfold.py:6](../../../../clinicadl/utils/split_manager/kfold.py#L6)

#### Signature

```python
class KFoldSplit(SplitManager):
    def __init__(
        self,
        caps_directory,
        tsv_path,
        diagnoses,
        n_splits,
        baseline=False,
        multi_cohort=False,
        split_list=None,
    ):
        ...
```

#### See also

- [SplitManager](./split_manager.md#splitmanager)

### KFoldSplit().allowed_splits_list

[Show source in kfold.py:31](../../../../clinicadl/utils/split_manager/kfold.py#L31)

#### Signature

```python
@property
def allowed_splits_list(self):
    ...
```

### KFoldSplit().max_length

[Show source in kfold.py:22](../../../../clinicadl/utils/split_manager/kfold.py#L22)

#### Signature

```python
def max_length(self) -> int:
    ...
```

### KFoldSplit().split_iterator

[Show source in kfold.py:35](../../../../clinicadl/utils/split_manager/kfold.py#L35)

#### Signature

```python
def split_iterator(self):
    ...
```