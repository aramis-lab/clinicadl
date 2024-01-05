# Utils

[Clinicadl Index](../../../README.md#clinicadl-index) /
[Clinicadl](../../index.md#clinicadl) /
[Quality Check](../index.md#quality-check) /
[T1 Linear](./index.md#t1-linear) /
Utils

> Auto-generated documentation for [clinicadl.quality_check.t1_linear.utils](../../../../clinicadl/quality_check/t1_linear/utils.py) module.

- [Utils](#utils)
  - [QCDataset](#qcdataset)
    - [QCDataset.get_padding](#qcdatasetget_padding)
    - [QCDataset.nii_transform](#qcdatasetnii_transform)
    - [QCDataset().pt_transform](#qcdataset()pt_transform)

## QCDataset

[Show source in utils.py:16](../../../../clinicadl/quality_check/t1_linear/utils.py#L16)

Dataset of MRI organized in a CAPS folder.

#### Signature

```python
class QCDataset(Dataset):
    def __init__(
        self,
        img_dir: Path,
        data_df,
        use_extracted_tensors=False,
        use_uncropped_image=True,
    ):
        ...
```

### QCDataset.get_padding

[Show source in utils.py:233](../../../../clinicadl/quality_check/t1_linear/utils.py#L233)

#### Signature

```python
@staticmethod
def get_padding(image):
    ...
```

### QCDataset.nii_transform

[Show source in utils.py:101](../../../../clinicadl/quality_check/t1_linear/utils.py#L101)

#### Signature

```python
@staticmethod
def nii_transform(image):
    ...
```

### QCDataset().pt_transform

[Show source in utils.py:169](../../../../clinicadl/quality_check/t1_linear/utils.py#L169)

#### Signature

```python
def pt_transform(self, image):
    ...
```