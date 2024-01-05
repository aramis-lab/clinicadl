# Generate

[Clinicadl Index](../../README.md#clinicadl-index) /
[Clinicadl](../index.md#clinicadl) /
[Generate](./index.md#generate) /
Generate

> Auto-generated documentation for [clinicadl.generate.generate](../../../clinicadl/generate/generate.py) module.

- [Generate](#generate)
  - [generate_hypometabolic_dataset](#generate_hypometabolic_dataset)
  - [generate_random_dataset](#generate_random_dataset)
  - [generate_shepplogan_dataset](#generate_shepplogan_dataset)
  - [generate_trivial_dataset](#generate_trivial_dataset)

## generate_hypometabolic_dataset

[Show source in generate.py:441](../../../clinicadl/generate/generate.py#L441)

Generates a dataset, based on the images of the CAPS directory, where all
the images are processed using a mask to generate a specific pathology.

Parameters
----------
caps_directory: Path
    Path to the CAPS directory.
output_dir: Path
    Folder containing the synthetic dataset in CAPS format.
n_subjects: int
    Number of subjects in each class of the synthetic dataset.
n_proc: int
    Number of cores used during the task.
tsv_path: Path
    Path to tsv file of list of subjects/sessions.
preprocessing: str
    Preprocessing performed. For now it must be 'pet-linear'.
pathology: str
    Name of the pathology to generate.
anomaly_degree: float
    Percentage of pathology applied.
sigma: int
    It is the parameter of the gaussian filter used for smoothing.
uncropped_image: bool
    If True the uncropped image of `t1-linear` or `pet-linear` will be used.

Returns
-------
Folder structure where images are stored in CAPS format.

Raises
------
IndexError: if `n_subjects` is higher than the length of the TSV file at `tsv_path`.

#### Signature

```python
def generate_hypometabolic_dataset(
    caps_directory: Path,
    output_dir: Path,
    n_subjects: int,
    n_proc: int,
    tsv_path: Optional[Path] = None,
    preprocessing: str = "pet-linear",
    pathology: str = "ad",
    anomaly_degree: float = 30,
    sigma: int = 5,
    uncropped_image: bool = False,
):
    ...
```



## generate_random_dataset

[Show source in generate.py:38](../../../clinicadl/generate/generate.py#L38)

Generates a random dataset.

Creates a random dataset for intractable classification task from the first
subject of the tsv file (other subjects/sessions different from the first
one are ignored. Degree of noise can be parameterized.

Parameters
----------
caps_directory: Path
    Path to the (input) CAPS directory.
output_dir: Path
    Folder containing the synthetic dataset in CAPS format.
n_subjects: int
    Number of subjects in each class of the synthetic dataset
tsv_path: Path
    Path to tsv file of list of subjects/sessions.
mean: float
    Mean of the gaussian noise
sigma: float
    Standard deviation of the gaussian noise
preprocessing: str
    Preprocessing performed. Must be in ['t1-linear', 't1-extensive'].
multi_cohort: bool
    If True caps_directory is the path to a TSV file linking cohort names and paths.
uncropped_image: bool
    If True the uncropped image of `t1-linear` or `pet-linear` will be used.
tracer: str
    name of the tracer when using `pet-linear` preprocessing.
suvr_reference_region: str
    name of the reference region when using `pet-linear` preprocessing.

Returns
-------
A folder written on the output_dir location (in CAPS format), also a
tsv file describing this output

#### Signature

```python
def generate_random_dataset(
    caps_directory: Path,
    output_dir: Path,
    n_subjects: int,
    tsv_path: Optional[Path] = None,
    mean: float = 0,
    sigma: float = 0.5,
    preprocessing: str = "t1-linear",
    multi_cohort: bool = False,
    uncropped_image: bool = False,
    tracer: Optional[str] = None,
    suvr_reference_region: Optional[str] = None,
):
    ...
```



## generate_shepplogan_dataset

[Show source in generate.py:335](../../../clinicadl/generate/generate.py#L335)

Creates a CAPS data set of synthetic data based on Shepp-Logan phantom.
Source NifTi files are not extracted, but directly the slices as tensors.

#### Arguments

- `output_dir` - path to the CAPS created.
- `img_size` - size of the square image.
- `labels_distribution` - gives the proportions of the three subtypes (ordered in a tuple) for each label.
- `extract_json` - name of the JSON file in which generation details are stored.
- `samples` - number of samples generated per class.
- `smoothing` - if True, an additional random smoothing is performed on top of all operations on each image.

#### Signature

```python
def generate_shepplogan_dataset(
    output_dir: Path,
    img_size: int,
    labels_distribution: Dict[str, Tuple[float, float, float]],
    extract_json: str = None,
    samples: int = 100,
    smoothing: bool = True,
):
    ...
```



## generate_trivial_dataset

[Show source in generate.py:161](../../../clinicadl/generate/generate.py#L161)

Generates a fully separable dataset.

Generates a dataset, based on the images of the CAPS directory, where a
half of the image is processed using a mask to occlude a specific region.
This procedure creates a dataset fully separable (images with half-right
processed and image with half-left processed)

Parameters
----------
caps_directory: Path
    Path to the CAPS directory.
output_dir: Path
    Folder containing the synthetic dataset in CAPS format.
n_subjects: int
    Number of subjects in each class of the synthetic dataset.
tsv_path: Path
    Path to tsv file of list of subjects/sessions.
preprocessing: str
    Preprocessing performed. Must be in ['linear', 'extensive'].
mask_path: Path
    Path to the extracted masks to generate the two labels.
atrophy_percent: float
    Percentage of atrophy applied.
multi_cohort: bool
    If True caps_directory is the path to a TSV file linking cohort names and paths.
uncropped_image: bool
    If True the uncropped image of `t1-linear` or `pet-linear` will be used.
tracer: str
    Name of the tracer when using `pet-linear` preprocessing.
suvr_reference_region: str
    Name of the reference region when using `pet-linear` preprocessing.

Returns
-------
    Folder structure where images are stored in CAPS format.

Raises
------
    IndexError: if `n_subjects` is higher than the length of the TSV file at `tsv_path`.

#### Signature

```python
def generate_trivial_dataset(
    caps_directory: Path,
    output_dir: Path,
    n_subjects: int,
    tsv_path: Optional[Path] = None,
    preprocessing: str = "t1-linear",
    mask_path: Optional[Path] = None,
    atrophy_percent: float = 60,
    multi_cohort: bool = False,
    uncropped_image: bool = False,
    tracer: str = "fdg",
    suvr_reference_region: str = "pons",
):
    ...
```