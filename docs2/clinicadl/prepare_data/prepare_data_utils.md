# Prepare Data Utils

[Clinicadl Index](../../README.md#clinicadl-index) /
[Clinicadl](../index.md#clinicadl) /
[Prepare Data](./index.md#prepare-data) /
Prepare Data Utils

> Auto-generated documentation for [clinicadl.prepare_data.prepare_data_utils](../../../clinicadl/prepare_data/prepare_data_utils.py) module.

- [Prepare Data Utils](#prepare-data-utils)
  - [check_mask_list](#check_mask_list)
  - [compute_discarded_slices](#compute_discarded_slices)
  - [compute_extract_json](#compute_extract_json)
  - [compute_folder_and_file_type](#compute_folder_and_file_type)
  - [compute_output_pattern](#compute_output_pattern)
  - [extract_images](#extract_images)
  - [extract_patch_path](#extract_patch_path)
  - [extract_patch_tensor](#extract_patch_tensor)
  - [extract_patches](#extract_patches)
  - [extract_roi](#extract_roi)
  - [extract_roi_path](#extract_roi_path)
  - [extract_roi_tensor](#extract_roi_tensor)
  - [extract_slice_path](#extract_slice_path)
  - [extract_slice_tensor](#extract_slice_tensor)
  - [extract_slices](#extract_slices)
  - [find_mask_path](#find_mask_path)
  - [get_parameters_dict](#get_parameters_dict)

## check_mask_list

[Show source in prepare_data_utils.py:331](../../../clinicadl/prepare_data/prepare_data_utils.py#L331)

#### Signature

```python
def check_mask_list(masks_location: Path, roi_list, mask_pattern, cropping):
    ...
```



## compute_discarded_slices

[Show source in prepare_data_utils.py:112](../../../clinicadl/prepare_data/prepare_data_utils.py#L112)

#### Signature

```python
def compute_discarded_slices(discarded_slices: Union[int, tuple]) -> Tuple[int, int]:
    ...
```



## compute_extract_json

[Show source in prepare_data_utils.py:54](../../../clinicadl/prepare_data/prepare_data_utils.py#L54)

#### Signature

```python
def compute_extract_json(extract_json: str) -> str:
    ...
```



## compute_folder_and_file_type

[Show source in prepare_data_utils.py:63](../../../clinicadl/prepare_data/prepare_data_utils.py#L63)

#### Signature

```python
def compute_folder_and_file_type(
    parameters: Dict[str, Any]
) -> Tuple[str, Dict[str, str]]:
    ...
```



## compute_output_pattern

[Show source in prepare_data_utils.py:389](../../../clinicadl/prepare_data/prepare_data_utils.py#L389)

Computes the output pattern of the region cropped (without the source file prefix)

#### Arguments

- `mask_path` - path to the masks
- `crop_output` - If True the output is cropped, and the descriptor CropRoi must exist

#### Returns

the output pattern

#### Signature

```python
def compute_output_pattern(mask_path: Path, crop_output):
    ...
```



## extract_images

[Show source in prepare_data_utils.py:305](../../../clinicadl/prepare_data/prepare_data_utils.py#L305)

Extract the images
This function convert nifti image to tensor (.pt) version of the image.
Tensor version is saved at the same location than input_img.

#### Arguments

- `input_img` - path to the NifTi input image.

#### Returns

- `filename` *str* - single tensor file  saved on the disk. Same location than input file.

#### Signature

```python
def extract_images(input_img: Path) -> List[Tuple[str, torch.Tensor]]:
    ...
```



## extract_patch_path

[Show source in prepare_data_utils.py:290](../../../clinicadl/prepare_data/prepare_data_utils.py#L290)

#### Signature

```python
def extract_patch_path(
    img_path: Path, patch_size: int, stride_size: int, patch_index: int
) -> str:
    ...
```



## extract_patch_tensor

[Show source in prepare_data_utils.py:267](../../../clinicadl/prepare_data/prepare_data_utils.py#L267)

Extracts a single patch from image_tensor

#### Signature

```python
def extract_patch_tensor(
    image_tensor: torch.Tensor,
    patch_size: int,
    stride_size: int,
    patch_index: int,
    patches_tensor: torch.Tensor = None,
) -> torch.Tensor:
    ...
```



## extract_patches

[Show source in prepare_data_utils.py:223](../../../clinicadl/prepare_data/prepare_data_utils.py#L223)

Extracts the patches
This function extracts patches form the preprocessed nifti image. Patch size
if provided as input and also the stride size. If stride size is smaller
than the patch size an overlap exist between consecutive patches. If stride
size is equal to path size there is no overlap. Otherwise, unprocessed
zones can exits.

#### Arguments

- `nii_path` - path to the NifTi input image.
- `patch_size` - size of a single patch.
- `stride_size` - size of the stride leading to next patch.

#### Returns

list of tuples containing the path to the extracted patch
    and the tensor of the corresponding patch.

#### Signature

```python
def extract_patches(
    nii_path: Path, patch_size: int, stride_size: int
) -> List[Tuple[str, torch.Tensor]]:
    ...
```



## extract_roi

[Show source in prepare_data_utils.py:424](../../../clinicadl/prepare_data/prepare_data_utils.py#L424)

Extracts regions of interest defined by masks
This function extracts regions of interest from preprocessed nifti images.
The regions are defined using binary masks that must be located in the CAPS
at `masks/tpl-<template>`.

#### Arguments

- `nii_path` - path to the NifTi input image.
- `masks_location` - path to the masks
- `mask_pattern` - pattern to identify the masks
- `cropped_input` - if the input is cropped or not (contains desc-Crop)
- `roi_names` - list of the names of the regions that will be extracted.
- `uncrop_output` - if True, the final region is not cropped.

#### Returns

list of tuples containing the path to the extracted ROI
    and the tensor of the corresponding ROI.

#### Signature

```python
def extract_roi(
    nii_path: Path,
    masks_location: Path,
    mask_pattern: str,
    cropped_input: bool,
    roi_names: List[str],
    uncrop_output: bool,
) -> List[Tuple[str, torch.Tensor]]:
    ...
```



## extract_roi_path

[Show source in prepare_data_utils.py:497](../../../clinicadl/prepare_data/prepare_data_utils.py#L497)

#### Signature

```python
def extract_roi_path(img_path: Path, mask_path: Path, uncrop_output: bool) -> str:
    ...
```



## extract_roi_tensor

[Show source in prepare_data_utils.py:468](../../../clinicadl/prepare_data/prepare_data_utils.py#L468)

#### Signature

```python
def extract_roi_tensor(
    image_tensor: torch.Tensor, mask_np, uncrop_output: bool
) -> torch.Tensor:
    ...
```



## extract_slice_path

[Show source in prepare_data_utils.py:199](../../../clinicadl/prepare_data/prepare_data_utils.py#L199)

#### Signature

```python
def extract_slice_path(
    img_path: Path, slice_direction: int, slice_mode: str, slice_index: int
) -> str:
    ...
```



## extract_slice_tensor

[Show source in prepare_data_utils.py:177](../../../clinicadl/prepare_data/prepare_data_utils.py#L177)

#### Signature

```python
def extract_slice_tensor(
    image_tensor: torch.Tensor, slice_direction: int, slice_mode: str, slice_index: int
) -> torch.Tensor:
    ...
```



## extract_slices

[Show source in prepare_data_utils.py:127](../../../clinicadl/prepare_data/prepare_data_utils.py#L127)

Extracts the slices from three directions
This function extracts slices form the preprocessed nifti image.

The direction of extraction can be defined either on sagittal direction (0),
coronal direction (1) or axial direction (other).

The output slices can be stored following two modes:
single (1 channel) or rgb (3 channels, all the same).

#### Arguments

- `nii_path` - path to the NifTi input image.
- `slice_direction` - along which axis slices are extracted.
- `slice_mode` - 'single' or 'rgb'.
- `discarded_slices` - Number of slices to discard at the beginning and the end of the image.
    Will be a tuple of two integers if the number of slices to discard at the beginning
    and at the end differ.

#### Returns

list of tuples containing the path to the extracted slice
    and the tensor of the corresponding slice.

#### Signature

```python
def extract_slices(
    nii_path: Path,
    slice_direction: int = 0,
    slice_mode: str = "single",
    discarded_slices: Union[int, tuple] = 0,
) -> List[Tuple[str, torch.Tensor]]:
    ...
```



## find_mask_path

[Show source in prepare_data_utils.py:349](../../../clinicadl/prepare_data/prepare_data_utils.py#L349)

Finds masks corresponding to the pattern asked and containing the adequate cropping description

#### Arguments

- `masks_location` - directory containing the masks.
- `roi` - name of the region.
- `mask_pattern` - pattern which should be found in the filename of the mask.
- `cropping` - if True the original image should contain the substring 'desc-Crop'.

#### Returns

path of the mask or None if nothing was found.
a human-friendly description of the pattern looked for.

#### Signature

```python
def find_mask_path(
    masks_location: Path, roi: str, mask_pattern: str, cropping: bool
) -> Tuple[str, str]:
    ...
```



## get_parameters_dict

[Show source in prepare_data_utils.py:10](../../../clinicadl/prepare_data/prepare_data_utils.py#L10)

#### Arguments

- `modality` - preprocessing procedure performed with Clinica.
- `extract_method` - mode of extraction (image, slice, patch, roi).
- `save_features` - If True modes are extracted, else images are extracted
    and the extraction of modes is done on-the-fly during training.
- `extract_json` - Name of the JSON file created to sum up the arguments of tensor extraction.
- `use_uncropped_image` - If True the cropped version of the image is used
    (specific to t1-linear and pet-linear).
- `custom_suffix` - string used to identify images when modality is custom.
- `tracer` - name of the tracer (specific to PET pipelines).
- `suvr_reference_region` - name of the reference region for normalization
    specific to PET pipelines)

#### Returns

The dictionary of parameters specific to the preprocessing

#### Signature

```python
def get_parameters_dict(
    modality: str,
    extract_method: str,
    save_features: bool,
    extract_json: str,
    use_uncropped_image: bool,
    custom_suffix: str,
    tracer: str,
    suvr_reference_region: str,
) -> Dict[str, Any]:
    ...
```