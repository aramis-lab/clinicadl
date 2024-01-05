# Generate Utils

[Clinicadl Index](../../README.md#clinicadl-index) /
[Clinicadl](../index.md#clinicadl) /
[Generate](./index.md#generate) /
Generate Utils

> Auto-generated documentation for [clinicadl.generate.generate_utils](../../../clinicadl/generate/generate_utils.py) module.

- [Generate Utils](#generate-utils)
  - [binary_t1_pgm](#binary_t1_pgm)
  - [find_file_type](#find_file_type)
  - [generate_scales](#generate_scales)
  - [generate_shepplogan_phantom](#generate_shepplogan_phantom)
  - [im_loss_roi_gaussian_distribution](#im_loss_roi_gaussian_distribution)
  - [load_and_check_tsv](#load_and_check_tsv)
  - [mask_processing](#mask_processing)
  - [write_missing_mods](#write_missing_mods)

## binary_t1_pgm

[Show source in generate_utils.py:101](../../../clinicadl/generate/generate_utils.py#L101)

#### Arguments

- `im_data` - probability gray maps

#### Returns

binarized probability gray maps

#### Signature

```python
def binary_t1_pgm(im_data: np.ndarray) -> np.ndarray:
    ...
```



## find_file_type

[Show source in generate_utils.py:17](../../../clinicadl/generate/generate_utils.py#L17)

#### Signature

```python
def find_file_type(
    preprocessing: str, uncropped_image: bool, tracer: str, suvr_reference_region: str
) -> Dict[str, str]:
    ...
```



## generate_scales

[Show source in generate_utils.py:158](../../../clinicadl/generate/generate_utils.py#L158)

#### Signature

```python
def generate_scales(size):
    ...
```



## generate_shepplogan_phantom

[Show source in generate_utils.py:169](../../../clinicadl/generate/generate_utils.py#L169)

Generate 2D Shepp-Logan phantom with random regions size. Phantoms also
simulate different kind of AD by generating smaller ROIs.

#### Arguments

- `img_size` - Size of the generated image (img_size x img_size).
- `label` - Take 0 or 1 or 2. Label of the generated image.
    If 0, the ROIs simulate a CN subject.
    If 1, the ROIs simulate type 1 of AD.
    if 2, the ROIs simulate type 2 of AD.
- `smoothing` - Default True. Apply Gaussian smoothing to the image.

#### Returns

- `img` - 2D Sheep Logan phantom with specified label.

#### Signature

```python
def generate_shepplogan_phantom(
    img_size: int, label: int = 0, smoothing: bool = True
) -> np.ndarray:
    ...
```



## im_loss_roi_gaussian_distribution

[Show source in generate_utils.py:114](../../../clinicadl/generate/generate_utils.py#L114)

Create a smooth atrophy in the input image on the region in the mask.
The value of the atrophy is computed with a Gaussian so it will appear smooth and
more realistic.

#### Arguments

- `im_data` - Input image that will be atrophied (obtained from a nifti file).
- `atlas_to_mask` - Binary mask of the region to atrophy.
- `min_value` - Percentage of atrophy between 0 and 100.

#### Returns

- `im_with_loss_gm_roi` - Image with atrophy in the specified ROI.

#### Signature

```python
def im_loss_roi_gaussian_distribution(
    im_data: np.ndarray, atlas_to_mask: np.ndarray, min_value: float
) -> np.ndarray:
    ...
```



## load_and_check_tsv

[Show source in generate_utils.py:58](../../../clinicadl/generate/generate_utils.py#L58)

#### Signature

```python
def load_and_check_tsv(
    tsv_path: Path, caps_dict: Dict[str, Path], output_path: Path
) -> pd.DataFrame:
    ...
```



## mask_processing

[Show source in generate_utils.py:322](../../../clinicadl/generate/generate_utils.py#L322)

#### Signature

```python
def mask_processing(mask, percentage, sigma):
    ...
```



## write_missing_mods

[Show source in generate_utils.py:43](../../../clinicadl/generate/generate_utils.py#L43)

#### Signature

```python
def write_missing_mods(output_dir: Path, output_df: pd.DataFrame):
    ...
```