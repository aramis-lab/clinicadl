# Quality Check

[Clinicadl Index](../../../README.md#clinicadl-index) /
[Clinicadl](../../index.md#clinicadl) /
[Quality Check](../index.md#quality-check) /
[T1 Linear](./index.md#t1-linear) /
Quality Check

> Auto-generated documentation for [clinicadl.quality_check.t1_linear.quality_check](../../../../clinicadl/quality_check/t1_linear/quality_check.py) module.

- [Quality Check](#quality-check)
  - [quality_check](#quality_check)

## quality_check

[Show source in quality_check.py:24](../../../../clinicadl/quality_check/t1_linear/quality_check.py#L24)

Performs t1-linear quality-check

Parameters
-----------
caps_dir: str (Path)
    Path to the input caps directory
output_path: str (Path)
    Path to the output TSV file.
tsv_path: str (Path)
    Path to the participant.tsv if the option was added.
threshold: float
    Threshold that indicates whether the image passes the quality check.
batch_size: int
n_proc: int
gpu: int
network: str
    Architecture of the pretrained network pretrained network that learned to classify images that are adequately registered.
    To chose between "darq" and "deep-qc"
use_tensor: bool
    To use tensor instead of nifti images
use_uncropped_image: bool
    To use uncropped images instead of the cropped ones.

#### Signature

```python
def quality_check(
    caps_dir: Path,
    output_path: Path,
    tsv_path: Path = None,
    threshold: float = 0.5,
    batch_size: int = 1,
    n_proc: int = 0,
    gpu: bool = True,
    network: str = "darq",
    use_tensor: bool = False,
    use_uncropped_image: bool = True,
):
    ...
```