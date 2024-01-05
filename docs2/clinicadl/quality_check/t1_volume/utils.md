# Utils

[Clinicadl Index](../../../README.md#clinicadl-index) /
[Clinicadl](../../index.md#clinicadl) /
[Quality Check](../index.md#quality-check) /
[T1 Volume](./index.md#t1-volume) /
Utils

> Auto-generated documentation for [clinicadl.quality_check.t1_volume.utils](../../../../clinicadl/quality_check/t1_volume/utils.py) module.

- [Utils](#utils)
  - [extract_metrics](#extract_metrics)
  - [nmi](#nmi)

## extract_metrics

[Show source in utils.py:13](../../../../clinicadl/quality_check/t1_volume/utils.py#L13)

#### Signature

```python
def extract_metrics(caps_dir: Path, output_dir: Path, group_label):
    ...
```



## nmi

[Show source in utils.py:113](../../../../clinicadl/quality_check/t1_volume/utils.py#L113)

Mutual information for joint histogram

#### Signature

```python
def nmi(occlusion1, occlusion2):
    ...
```