# Prepare Data Cli

[Clinicadl Index](../../README.md#clinicadl-index) /
[Clinicadl](../index.md#clinicadl) /
[Prepare Data](./index.md#prepare-data) /
Prepare Data Cli

> Auto-generated documentation for [clinicadl.prepare_data.prepare_data_cli](../../../clinicadl/prepare_data/prepare_data_cli.py) module.

- [Prepare Data Cli](#prepare-data-cli)
  - [RegistrationOrderGroup](#registrationordergroup)
    - [RegistrationOrderGroup().list_commands](#registrationordergroup()list_commands)
  - [cli](#cli)
  - [image_cli](#image_cli)
  - [patch_cli](#patch_cli)
  - [roi_cli](#roi_cli)
  - [slice_cli](#slice_cli)

## RegistrationOrderGroup

[Show source in prepare_data_cli.py:294](../../../clinicadl/prepare_data/prepare_data_cli.py#L294)

CLI group which lists commands by order or registration.

#### Signature

```python
class RegistrationOrderGroup(click.Group):
    ...
```

### RegistrationOrderGroup().list_commands

[Show source in prepare_data_cli.py:297](../../../clinicadl/prepare_data/prepare_data_cli.py#L297)

#### Signature

```python
def list_commands(self, ctx):
    ...
```



## cli

[Show source in prepare_data_cli.py:301](../../../clinicadl/prepare_data/prepare_data_cli.py#L301)

Extract Pytorch tensors from nifti images.

#### Signature

```python
@click.group(cls=RegistrationOrderGroup, name="prepare-data", no_args_is_help=True)
def cli() -> None:
    ...
```



## image_cli

[Show source in prepare_data_cli.py:12](../../../clinicadl/prepare_data/prepare_data_cli.py#L12)

Extract image from nifti images.

CAPS_DIRECTORY is the CAPS folder where nifti images are stored and tensor will be saved.

MODALITY [t1-linear|pet-linear|custom] is the clinica pipeline name used for image preprocessing.

#### Signature

```python
@click.command(name="image", no_args_is_help=True)
@cli_param.argument.caps_directory
@cli_param.argument.modality
@cli_param.option.n_proc
@cli_param.option.subjects_sessions_tsv
@cli_param.option.extract_json
@cli_param.option.use_uncropped_image
@cli_param.option.tracer
@cli_param.option.suvr_reference_region
@cli_param.option.custom_suffix
def image_cli(
    caps_directory: Path,
    modality: str,
    n_proc: int,
    subjects_sessions_tsv: Optional[Path] = None,
    extract_json: str = None,
    use_uncropped_image: bool = False,
    tracer: Optional[str] = None,
    suvr_reference_region: Optional[str] = None,
    custom_suffix: str = "",
):
    ...
```



## patch_cli

[Show source in prepare_data_cli.py:57](../../../clinicadl/prepare_data/prepare_data_cli.py#L57)

Extract patch from nifti images.

CAPS_DIRECTORY is the CAPS folder where nifti images are stored and tensor will be saved.

MODALITY [t1-linear|pet-linear|custom] is the clinica pipeline name used for image preprocessing.

#### Signature

```python
@click.command(name="patch", no_args_is_help=True)
@cli_param.argument.caps_directory
@cli_param.argument.modality
@cli_param.option.n_proc
@cli_param.option.save_features
@cli_param.option.subjects_sessions_tsv
@cli_param.option.extract_json
@cli_param.option.use_uncropped_image
@click.option("-ps", "--patch_size", default=50, show_default=True, help="Patch size.")
@click.option("-ss", "--stride_size", default=50, show_default=True, help="Stride size.")
@cli_param.option.tracer
@cli_param.option.suvr_reference_region
@cli_param.option.custom_suffix
def patch_cli(
    caps_directory: Path,
    modality: str,
    n_proc: int,
    save_features: bool = False,
    subjects_sessions_tsv: Optional[Path] = None,
    extract_json: str = None,
    use_uncropped_image: bool = False,
    patch_size: int = 50,
    stride_size: int = 50,
    tracer: Optional[str] = None,
    suvr_reference_region: Optional[str] = None,
    custom_suffix: str = "",
):
    ...
```



## roi_cli

[Show source in prepare_data_cli.py:206](../../../clinicadl/prepare_data/prepare_data_cli.py#L206)

Extract roi from nifti images.

CAPS_DIRECTORY is the CAPS folder where nifti images are stored and tensor will be saved.

MODALITY [t1-linear|pet-linear|custom] is the clinica pipeline name used for image preprocessing.

#### Signature

```python
@click.command(name="roi", no_args_is_help=True)
@cli_param.argument.caps_directory
@cli_param.argument.modality
@cli_param.option.n_proc
@cli_param.option.save_features
@cli_param.option.subjects_sessions_tsv
@cli_param.option.extract_json
@cli_param.option.use_uncropped_image
@click.option(
    "--roi_list",
    type=str,
    required=True,
    multiple=True,
    help="List of regions to be extracted",
)
@click.option(
    "--roi_uncrop_output",
    type=bool,
    default=False,
    is_flag=True,
    help=(
        "Disable cropping option so the output tensors have the same size than the whole"
        " image."
    ),
)
@click.option(
    "--roi_custom_template",
    "-ct",
    type=str,
    default="",
    help=(
        "Template name if MODALITY is `custom`.\n        Name of the template used for"
        " registration during the preprocessing procedure."
    ),
)
@click.option(
    "--roi_custom_mask_pattern",
    "-cmp",
    type=str,
    default="",
    help=(
        "Mask pattern if MODALITY is `custom`.\n            If given will select only"
        " the masks containing the string given.\n            The mask with the shortest"
        " name is taken."
    ),
)
@cli_param.option.tracer
@cli_param.option.suvr_reference_region
@cli_param.option.custom_suffix
def roi_cli(
    caps_directory: Path,
    modality: str,
    n_proc: int,
    save_features: bool = False,
    subjects_sessions_tsv: Optional[Path] = None,
    extract_json: str = None,
    use_uncropped_image: bool = False,
    roi_list: list = [],
    roi_uncrop_output: bool = False,
    roi_custom_template: str = "",
    roi_custom_mask_pattern: str = "",
    tracer: Optional[str] = None,
    suvr_reference_region: Optional[str] = None,
    custom_suffix: str = "",
):
    ...
```



## slice_cli

[Show source in prepare_data_cli.py:123](../../../clinicadl/prepare_data/prepare_data_cli.py#L123)

Extract slice from nifti images.

CAPS_DIRECTORY is the CAPS folder where nifti images are stored and tensor will be saved.

MODALITY [t1-linear|pet-linear|custom] is the clinica pipeline name used for image preprocessing.

#### Signature

```python
@click.command(name="slice", no_args_is_help=True)
@cli_param.argument.caps_directory
@cli_param.argument.modality
@cli_param.option.n_proc
@cli_param.option.save_features
@cli_param.option.subjects_sessions_tsv
@cli_param.option.extract_json
@cli_param.option.use_uncropped_image
@click.option(
    "-sd",
    "--slice_direction",
    type=click.IntRange(0, 2),
    default=0,
    show_default=True,
    help="Slice direction. 0: Sagittal plane, 1: Coronal plane, 2: Axial plane.",
)
@click.option(
    "-sm",
    "--slice_mode",
    type=click.Choice(["rgb", "single"]),
    default="rgb",
    show_default=True,
    help=(
        "rgb: Save the slice in three identical channels, single: Save the slice in a"
        " single channel."
    ),
)
@click.option(
    "-ds",
    "--discarded_slices",
    type=int,
    default=(0, 0),
    multiple=2,
    help=(
        "Number of slices discarded from respectively the beginning and\n        the end"
        " of the MRI volume.  If only one argument is given, it will be\n        used"
        " for both sides."
    ),
)
@cli_param.option.tracer
@cli_param.option.suvr_reference_region
@cli_param.option.custom_suffix
def slice_cli(
    caps_directory: Path,
    modality: str,
    n_proc: int,
    save_features: bool = False,
    subjects_sessions_tsv: Optional[Path] = None,
    extract_json: str = None,
    use_uncropped_image: bool = False,
    slice_direction: int = 0,
    slice_mode: str = "rgb",
    discarded_slices: int = 0,
    tracer: Optional[str] = None,
    suvr_reference_region: Optional[str] = None,
    custom_suffix: str = "",
):
    ...
```