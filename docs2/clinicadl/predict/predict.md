# Predict

[Clinicadl Index](../../README.md#clinicadl-index) /
[Clinicadl](../index.md#clinicadl) /
[Predict](./index.md#predict) /
Predict

> Auto-generated documentation for [clinicadl.predict.predict](../../../clinicadl/predict/predict.py) module.

- [Predict](#predict)
  - [predict](#predict)

## predict

[Show source in predict.py:9](../../../clinicadl/predict/predict.py#L9)

This function loads a MAPS and predicts the global metrics and individual values
for all the models selected using a metric in selection_metrics.

#### Arguments

- `maps_dir` - path to the MAPS.
- `data_group` - name of the data group tested.
- `caps_directory` - path to the CAPS folder. For more information please refer to
    [clinica documentation](https://aramislab.paris.inria.fr/clinica/docs/public/latest/CAPS/Introduction/).
- `tsv_path` - path to a TSV file containing the list of participants and sessions to interpret.
- `use_labels` - by default is True. If False no metrics tsv files will be written.
- `label` - Name of the target value, if different from training.
- `gpu` - if true, it uses gpu.
- `n_proc` - num_workers used in DataLoader
- `batch_size` - batch size of the DataLoader
- `selection_metrics` - list of metrics to find best models to be evaluated.
- `diagnoses` - list of diagnoses to be tested if tsv_path is a folder.
- `multi_cohort` - If True caps_directory is the path to a TSV file linking cohort names and paths.
- `overwrite` - If True former definition of data group is erased
- `save_tensor` - For reconstruction task only, if True it will save the reconstruction as .pt file in the MAPS.
- `save_nifti` - For reconstruction task only, if True it will save the reconstruction as NIfTI file in the MAPS.

#### Signature

```python
def predict(
    maps_dir: Path,
    data_group: str,
    caps_directory: Path,
    tsv_path: Path,
    use_labels: bool = True,
    label: str = None,
    gpu: bool = True,
    n_proc: int = 0,
    batch_size: int = 1,
    selection_metrics: List[str] = None,
    diagnoses: List[str] = None,
    multi_cohort: bool = False,
    overwrite: bool = False,
    save_tensor: bool = False,
    save_nifti: bool = False,
):
    ...
```