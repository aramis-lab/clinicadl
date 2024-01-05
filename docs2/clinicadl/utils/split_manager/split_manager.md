# SplitManager

[Clinicadl Index](../../../README.md#clinicadl-index) /
[Clinicadl](../../index.md#clinicadl) /
[Utils](../index.md#utils) /
[Split Manager](./index.md#split-manager) /
SplitManager

> Auto-generated documentation for [clinicadl.utils.split_manager.split_manager](../../../../clinicadl/utils/split_manager/split_manager.py) module.

- [SplitManager](#splitmanager)
  - [SplitManager](#splitmanager-1)
    - [SplitManager().__getitem__](#splitmanager()__getitem__)
    - [SplitManager().allowed_splits_list](#splitmanager()allowed_splits_list)
    - [SplitManager().concatenate_diagnoses](#splitmanager()concatenate_diagnoses)
    - [SplitManager().max_length](#splitmanager()max_length)
    - [SplitManager().split_iterator](#splitmanager()split_iterator)

## SplitManager

[Show source in split_manager.py:17](../../../../clinicadl/utils/split_manager/split_manager.py#L17)

#### Signature

```python
class SplitManager:
    def __init__(
        self,
        caps_directory: Path,
        tsv_path: Path,
        diagnoses,
        baseline=False,
        multi_cohort=False,
        split_list=None,
    ):
        ...
```

### SplitManager().__getitem__

[Show source in split_manager.py:71](../../../../clinicadl/utils/split_manager/split_manager.py#L71)

Returns a dictionary of DataFrames with train and validation data.

#### Arguments

- `item` *int* - Index of the split wanted.

#### Returns

- `Dict[str:pd.DataFrame]` - dictionary with two keys (train and validation).

#### Signature

```python
def __getitem__(self, item):
    ...
```

### SplitManager().allowed_splits_list

[Show source in split_manager.py:60](../../../../clinicadl/utils/split_manager/split_manager.py#L60)

List of possible splits if no restriction was applied

#### Returns

- `list[int]` - list of all possible splits

#### Signature

```python
@property
@abc.abstractmethod
def allowed_splits_list(self):
    ...
```

### SplitManager().concatenate_diagnoses

[Show source in split_manager.py:124](../../../../clinicadl/utils/split_manager/split_manager.py#L124)

Concatenated the diagnoses needed to form the train and validation sets.

#### Signature

```python
def concatenate_diagnoses(self, split, cohort_path: Path = None, cohort_diagnoses=None):
    ...
```

### SplitManager().max_length

[Show source in split_manager.py:51](../../../../clinicadl/utils/split_manager/split_manager.py#L51)

Maximum number of splits

#### Signature

```python
@abc.abstractmethod
def max_length(self) -> int:
    ...
```

### SplitManager().split_iterator

[Show source in split_manager.py:218](../../../../clinicadl/utils/split_manager/split_manager.py#L218)

Returns an iterable to iterate on all splits wanted.

#### Signature

```python
@abc.abstractmethod
def split_iterator(self):
    ...
```