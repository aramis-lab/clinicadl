# Getter

[Clinicadl Index](../../../README.md#clinicadl-index) /
[Clinicadl](../../index.md#clinicadl) /
[Utils](../index.md#utils) /
[Meta Maps](./index.md#meta-maps) /
Getter

> Auto-generated documentation for [clinicadl.utils.meta_maps.getter](../../../../clinicadl/utils/meta_maps/getter.py) module.

- [Getter](#getter)
  - [meta_maps_analysis](#meta_maps_analysis)

## meta_maps_analysis

[Show source in getter.py:12](../../../../clinicadl/utils/meta_maps/getter.py#L12)

This function summarizes the validation performance according to `evaluation_metric`
of several MAPS stored in the folder `launch_dir`.
The output TSV files are written in `launch_dir`.

#### Arguments

- `launch_dir` *str* - Path to the directory containing several MAPS.
- `evaluation_metric` *str* - Name of the metric used for validation evaluation.

#### Signature

```python
def meta_maps_analysis(launch_dir: Path, evaluation_metric="loss"):
    ...
```