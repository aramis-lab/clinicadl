# SingleSplit

[Clinicadl Index](../../../README.md#clinicadl-index) /
[Clinicadl](../../index.md#clinicadl) /
[Utils](../index.md#utils) /
[Split Manager](./index.md#split-manager) /
SingleSplit

> Auto-generated documentation for [clinicadl.utils.split_manager.single_split](../../../../clinicadl/utils/split_manager/single_split.py) module.

- [SingleSplit](#singlesplit)
  - [SingleSplit](#singlesplit-1)
    - [SingleSplit().allowed_splits_list](#singlesplit()allowed_splits_list)
    - [SingleSplit().max_length](#singlesplit()max_length)
    - [SingleSplit().split_iterator](#singlesplit()split_iterator)

## SingleSplit

[Show source in single_split.py:6](../../../../clinicadl/utils/split_manager/single_split.py#L6)

#### Signature

```python
class SingleSplit(SplitManager):
    def __init__(
        self,
        caps_directory,
        tsv_path,
        diagnoses,
        baseline=False,
        multi_cohort=False,
        split_list=None,
    ):
        ...
```

#### See also

- [SplitManager](./split_manager.md#splitmanager)

### SingleSplit().allowed_splits_list

[Show source in single_split.py:26](../../../../clinicadl/utils/split_manager/single_split.py#L26)

#### Signature

```python
@property
def allowed_splits_list(self):
    ...
```

### SingleSplit().max_length

[Show source in single_split.py:20](../../../../clinicadl/utils/split_manager/single_split.py#L20)

#### Signature

```python
def max_length(self) -> int:
    ...
```

### SingleSplit().split_iterator

[Show source in single_split.py:30](../../../../clinicadl/utils/split_manager/single_split.py#L30)

#### Signature

```python
def split_iterator(self):
    ...
```