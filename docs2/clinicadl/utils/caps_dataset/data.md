# Data

[Clinicadl Index](../../../README.md#clinicadl-index) /
[Clinicadl](../../index.md#clinicadl) /
[Utils](../index.md#utils) /
[Caps Dataset](./index.md#caps-dataset) /
Data

> Auto-generated documentation for [clinicadl.utils.caps_dataset.data](../../../../clinicadl/utils/caps_dataset/data.py) module.

- [Data](#data)
  - [CapsDataset](#capsdataset)
    - [CapsDataset().__getitem__](#capsdataset()__getitem__)
    - [CapsDataset.create_caps_dict](#capsdatasetcreate_caps_dict)
    - [CapsDataset().elem_index](#capsdataset()elem_index)
    - [CapsDataset().eval](#capsdataset()eval)
    - [CapsDataset().label_fn](#capsdataset()label_fn)
    - [CapsDataset().num_elem_per_image](#capsdataset()num_elem_per_image)
    - [CapsDataset().train](#capsdataset()train)
  - [CapsDatasetImage](#capsdatasetimage)
    - [CapsDatasetImage().elem_index](#capsdatasetimage()elem_index)
    - [CapsDatasetImage().num_elem_per_image](#capsdatasetimage()num_elem_per_image)
  - [CapsDatasetPatch](#capsdatasetpatch)
    - [CapsDatasetPatch().elem_index](#capsdatasetpatch()elem_index)
    - [CapsDatasetPatch().num_elem_per_image](#capsdatasetpatch()num_elem_per_image)
  - [CapsDatasetRoi](#capsdatasetroi)
    - [CapsDatasetRoi().elem_index](#capsdatasetroi()elem_index)
    - [CapsDatasetRoi().num_elem_per_image](#capsdatasetroi()num_elem_per_image)
  - [CapsDatasetSlice](#capsdatasetslice)
    - [CapsDatasetSlice().elem_index](#capsdatasetslice()elem_index)
    - [CapsDatasetSlice().num_elem_per_image](#capsdatasetslice()num_elem_per_image)
  - [GaussianSmoothing](#gaussiansmoothing)
  - [MinMaxNormalization](#minmaxnormalization)
  - [NanRemoval](#nanremoval)
  - [RandomCropPad](#randomcroppad)
  - [RandomNoising](#randomnoising)
  - [RandomSmoothing](#randomsmoothing)
  - [ToTensor](#totensor)
  - [check_multi_cohort_tsv](#check_multi_cohort_tsv)
  - [get_transforms](#get_transforms)
  - [load_data_test](#load_data_test)
  - [load_data_test_single](#load_data_test_single)
  - [return_dataset](#return_dataset)

## CapsDataset

[Show source in data.py:40](../../../../clinicadl/utils/caps_dataset/data.py#L40)

Abstract class for all derived CapsDatasets.

#### Signature

```python
class CapsDataset(Dataset):
    def __init__(
        self,
        caps_directory: Path,
        data_df: pd.DataFrame,
        preprocessing_dict: Dict[str, Any],
        transformations: Optional[Callable],
        label_presence: bool,
        label: str = None,
        label_code: Dict[Any, int] = None,
        augmentation_transformations: Optional[Callable] = None,
        multi_cohort: bool = False,
    ):
        ...
```

### CapsDataset().__getitem__

[Show source in data.py:242](../../../../clinicadl/utils/caps_dataset/data.py#L242)

Gets the sample containing all the information needed for training and testing tasks.

#### Arguments

- `idx` - row number of the meta-data contained in self.df

#### Returns

dictionary with following items:
    - "image" (torch.Tensor): the input given to the model,
    - "label" (int or float): the label used in criterion,
    - "participant_id" (str): ID of the participant,
    - "session_id" (str): ID of the session,
    - f"{self.mode}_id" (int): number of the element,
    - `-` *"image_path"* - path to the image loaded in CAPS.

#### Signature

```python
@abc.abstractmethod
def __getitem__(self, idx: int) -> Dict[str, Any]:
    ...
```

### CapsDataset.create_caps_dict

[Show source in data.py:114](../../../../clinicadl/utils/caps_dataset/data.py#L114)

#### Signature

```python
@staticmethod
def create_caps_dict(caps_directory: Path, multi_cohort: bool) -> Dict[str, Path]:
    ...
```

### CapsDataset().elem_index

[Show source in data.py:87](../../../../clinicadl/utils/caps_dataset/data.py#L87)

#### Signature

```python
@property
@abc.abstractmethod
def elem_index(self):
    ...
```

### CapsDataset().eval

[Show source in data.py:266](../../../../clinicadl/utils/caps_dataset/data.py#L266)

Put the dataset on evaluation mode (data augmentation is not performed).

#### Signature

```python
def eval(self):
    ...
```

### CapsDataset().label_fn

[Show source in data.py:92](../../../../clinicadl/utils/caps_dataset/data.py#L92)

Returns the label value usable in criterion.

#### Arguments

- `target` - value of the target.

#### Returns

- `label` - value of the label usable in criterion.

#### Signature

```python
def label_fn(self, target: Union[str, float, int]) -> Union[float, int]:
    ...
```

### CapsDataset().num_elem_per_image

[Show source in data.py:261](../../../../clinicadl/utils/caps_dataset/data.py#L261)

Computes the number of elements per image based on the full image.

#### Signature

```python
@abc.abstractmethod
def num_elem_per_image(self) -> int:
    ...
```

### CapsDataset().train

[Show source in data.py:271](../../../../clinicadl/utils/caps_dataset/data.py#L271)

Put the dataset on training mode (data augmentation is performed).

#### Signature

```python
def train(self):
    ...
```



## CapsDatasetImage

[Show source in data.py:277](../../../../clinicadl/utils/caps_dataset/data.py#L277)

Dataset of MRI organized in a CAPS folder.

#### Signature

```python
class CapsDatasetImage(CapsDataset):
    def __init__(
        self,
        caps_directory: Path,
        data_file: pd.DataFrame,
        preprocessing_dict: Dict[str, Any],
        train_transformations: Optional[Callable] = None,
        label_presence: bool = True,
        label: str = None,
        label_code: Dict[str, int] = None,
        all_transformations: Optional[Callable] = None,
        multi_cohort: bool = False,
    ):
        ...
```

#### See also

- [CapsDataset](#capsdataset)

### CapsDatasetImage().elem_index

[Show source in data.py:320](../../../../clinicadl/utils/caps_dataset/data.py#L320)

#### Signature

```python
@property
def elem_index(self):
    ...
```

### CapsDatasetImage().num_elem_per_image

[Show source in data.py:347](../../../../clinicadl/utils/caps_dataset/data.py#L347)

#### Signature

```python
def num_elem_per_image(self):
    ...
```



## CapsDatasetPatch

[Show source in data.py:351](../../../../clinicadl/utils/caps_dataset/data.py#L351)

#### Signature

```python
class CapsDatasetPatch(CapsDataset):
    def __init__(
        self,
        caps_directory: Path,
        data_file: pd.DataFrame,
        preprocessing_dict: Dict[str, Any],
        train_transformations: Optional[Callable] = None,
        patch_index: Optional[int] = None,
        label_presence: bool = True,
        label: str = None,
        label_code: Dict[str, int] = None,
        all_transformations: Optional[Callable] = None,
        multi_cohort: bool = False,
    ):
        ...
```

#### See also

- [CapsDataset](#capsdataset)

### CapsDatasetPatch().elem_index

[Show source in data.py:397](../../../../clinicadl/utils/caps_dataset/data.py#L397)

#### Signature

```python
@property
def elem_index(self):
    ...
```

### CapsDatasetPatch().num_elem_per_image

[Show source in data.py:436](../../../../clinicadl/utils/caps_dataset/data.py#L436)

#### Signature

```python
def num_elem_per_image(self):
    ...
```



## CapsDatasetRoi

[Show source in data.py:455](../../../../clinicadl/utils/caps_dataset/data.py#L455)

#### Signature

```python
class CapsDatasetRoi(CapsDataset):
    def __init__(
        self,
        caps_directory: Path,
        data_file: pd.DataFrame,
        preprocessing_dict: Dict[str, Any],
        roi_index: Optional[int] = None,
        train_transformations: Optional[Callable] = None,
        label_presence: bool = True,
        label: str = None,
        label_code: Dict[str, int] = None,
        all_transformations: Optional[Callable] = None,
        multi_cohort: bool = False,
    ):
        ...
```

#### See also

- [CapsDataset](#capsdataset)

### CapsDatasetRoi().elem_index

[Show source in data.py:504](../../../../clinicadl/utils/caps_dataset/data.py#L504)

#### Signature

```python
@property
def elem_index(self):
    ...
```

### CapsDatasetRoi().num_elem_per_image

[Show source in data.py:547](../../../../clinicadl/utils/caps_dataset/data.py#L547)

#### Signature

```python
def num_elem_per_image(self):
    ...
```



## CapsDatasetSlice

[Show source in data.py:617](../../../../clinicadl/utils/caps_dataset/data.py#L617)

#### Signature

```python
class CapsDatasetSlice(CapsDataset):
    def __init__(
        self,
        caps_directory: Path,
        data_file: pd.DataFrame,
        preprocessing_dict: Dict[str, Any],
        slice_index: Optional[int] = None,
        train_transformations: Optional[Callable] = None,
        label_presence: bool = True,
        label: str = None,
        label_code: Dict[str, int] = None,
        all_transformations: Optional[Callable] = None,
        multi_cohort: bool = False,
    ):
        ...
```

#### See also

- [CapsDataset](#capsdataset)

### CapsDatasetSlice().elem_index

[Show source in data.py:669](../../../../clinicadl/utils/caps_dataset/data.py#L669)

#### Signature

```python
@property
def elem_index(self):
    ...
```

### CapsDatasetSlice().num_elem_per_image

[Show source in data.py:710](../../../../clinicadl/utils/caps_dataset/data.py#L710)

#### Signature

```python
def num_elem_per_image(self):
    ...
```



## GaussianSmoothing

[Show source in data.py:874](../../../../clinicadl/utils/caps_dataset/data.py#L874)

#### Signature

```python
class GaussianSmoothing(object):
    def __init__(self, sigma):
        ...
```



## MinMaxNormalization

[Show source in data.py:899](../../../../clinicadl/utils/caps_dataset/data.py#L899)

Normalizes a tensor between 0 and 1

#### Signature

```python
class MinMaxNormalization(object):
    ...
```



## NanRemoval

[Show source in data.py:906](../../../../clinicadl/utils/caps_dataset/data.py#L906)

#### Signature

```python
class NanRemoval(object):
    def __init__(self):
        ...
```



## RandomCropPad

[Show source in data.py:852](../../../../clinicadl/utils/caps_dataset/data.py#L852)

#### Signature

```python
class RandomCropPad(object):
    def __init__(self, length):
        ...
```



## RandomNoising

[Show source in data.py:821](../../../../clinicadl/utils/caps_dataset/data.py#L821)

Applies a random zoom to a tensor

#### Signature

```python
class RandomNoising(object):
    def __init__(self, sigma=0.1):
        ...
```



## RandomSmoothing

[Show source in data.py:835](../../../../clinicadl/utils/caps_dataset/data.py#L835)

Applies a random zoom to a tensor

#### Signature

```python
class RandomSmoothing(object):
    def __init__(self, sigma=1):
        ...
```



## ToTensor

[Show source in data.py:889](../../../../clinicadl/utils/caps_dataset/data.py#L889)

Convert image type to Tensor and diagnosis to diagnosis code

#### Signature

```python
class ToTensor(object):
    ...
```



## check_multi_cohort_tsv

[Show source in data.py:1067](../../../../clinicadl/utils/caps_dataset/data.py#L1067)

Checks that a multi-cohort TSV file is valid.

#### Arguments

- `tsv_df` *pd.DataFrame* - DataFrame of multi-cohort definition.
- `purpose` *str* - what the TSV file describes (CAPS or TSV).

#### Raises

- `ValueError` - if the TSV file is badly formatted.

#### Signature

```python
def check_multi_cohort_tsv(tsv_df, purpose):
    ...
```



## get_transforms

[Show source in data.py:922](../../../../clinicadl/utils/caps_dataset/data.py#L922)

Outputs the transformations that will be applied to the dataset

#### Arguments

- `normalize` - if True will perform MinMaxNormalization.
- `data_augmentation` - list of data augmentation performed on the training set.

#### Returns

transforms to apply in train and evaluation mode / transforms to apply in evaluation mode only.

#### Signature

```python
def get_transforms(
    normalize: bool = True, data_augmentation: List[str] = None
) -> Tuple[transforms.Compose, transforms.Compose]:
    ...
```



## load_data_test

[Show source in data.py:963](../../../../clinicadl/utils/caps_dataset/data.py#L963)

Load data not managed by split_manager.

#### Arguments

- `test_path` *str* - path to the test TSV files / split directory / TSV file for multi-cohort
- `diagnoses_list` *List[str]* - list of the diagnoses wanted in case of split_dir or multi-cohort
- `baseline` *bool* - If True baseline sessions only used (split_dir handling only).
- `multi_cohort` *bool* - If True considers multi-cohort setting.

#### Signature

```python
def load_data_test(test_path: Path, diagnoses_list, baseline=True, multi_cohort=False):
    ...
```



## load_data_test_single

[Show source in data.py:1022](../../../../clinicadl/utils/caps_dataset/data.py#L1022)

#### Signature

```python
def load_data_test_single(test_path: Path, diagnoses_list, baseline=True):
    ...
```



## return_dataset

[Show source in data.py:725](../../../../clinicadl/utils/caps_dataset/data.py#L725)

Return appropriate Dataset according to given options.

#### Arguments

- `input_dir` - path to a directory containing a CAPS structure.
- `data_df` - List subjects, sessions and diagnoses.
- `preprocessing_dict` - preprocessing dict contained in the JSON file of prepare_data.
- `train_transformations` - Optional transform to be applied during training only.
- `all_transformations` - Optional transform to be applied during training and evaluation.
- `label` - Name of the column in data_df containing the label.
- `label_code` - label code that links the output node number to label value.
- `cnn_index` - Index of the CNN in a multi-CNN paradigm (optional).
- `label_presence` - If True the diagnosis will be extracted from the given DataFrame.
- `multi_cohort` - If True caps_directory is the path to a TSV file linking cohort names and paths.

#### Returns

the corresponding dataset.

#### Signature

```python
def return_dataset(
    input_dir: Path,
    data_df: pd.DataFrame,
    preprocessing_dict: Dict[str, Any],
    all_transformations: Optional[Callable],
    label: str = None,
    label_code: Dict[str, int] = None,
    train_transformations: Optional[Callable] = None,
    cnn_index: int = None,
    label_presence: bool = True,
    multi_cohort: bool = False,
) -> CapsDataset:
    ...
```

#### See also

- [CapsDataset](#capsdataset)