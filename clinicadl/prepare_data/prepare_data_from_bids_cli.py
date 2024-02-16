from pathlib import Path
from typing import Optional

import click

from clinicadl.utils import cli_param

from .prepare_data import DeepLearningPrepareData
from .prepare_data_utils import get_parameters_dict


@click.command(name="image", no_args_is_help=True)
@cli_param.argument.bids_directory
@cli_param.argument.caps_directory
@cli_param.argument.modality_bids
@cli_param.option.n_proc
@cli_param.option.subjects_sessions_tsv
@cli_param.option.extract_json
@cli_param.option.use_uncropped_image
@cli_param.option.tracer
@cli_param.option.suvr_reference_region
@cli_param.option.custom_suffix
def image_bids_cli(
    bids_directory: Path,
    caps_directory: Path,
    modality_bids: str,
    n_proc: int,
    subjects_sessions_tsv: Optional[Path] = None,
    extract_json: str = None,
    use_uncropped_image: bool = False,
    tracer: Optional[str] = None,
    suvr_reference_region: Optional[str] = None,
    custom_suffix: str = "",
):
    """Extract image from nifti images.

    CAPS_DIRECTORY is the CAPS folder where nifti images are stored and tensor will be saved.

    MODALITY [t1-linear|pet-linear|custom] is the clinica pipeline name used for image preprocessing.
    """
    parameters = get_parameters_dict(
        modality_bids,
        "image",
        False,
        extract_json,
        use_uncropped_image,
        custom_suffix,
        tracer,
        suvr_reference_region,
    )
    DeepLearningPrepareData(
        caps_directory=caps_directory,
        tsv_file=subjects_sessions_tsv,
        n_proc=n_proc,
        parameters=parameters,
        from_bids=bids_directory,
    )


@click.command(name="patch", no_args_is_help=True)
@cli_param.argument.bids_directory
@cli_param.argument.caps_directory
@cli_param.argument.modality_bids
@cli_param.option.n_proc
@cli_param.option.save_features
@cli_param.option.subjects_sessions_tsv
@cli_param.option.extract_json
@cli_param.option.use_uncropped_image
@click.option(
    "-ps",
    "--patch_size",
    default=50,
    show_default=True,
    help="Patch size.",
)
@click.option(
    "-ss",
    "--stride_size",
    default=50,
    show_default=True,
    help="Stride size.",
)
@cli_param.option.tracer
@cli_param.option.suvr_reference_region
@cli_param.option.custom_suffix
def patch_bids_cli(
    bids_directory: Path,
    caps_directory: Path,
    modality_bids: str,
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
    """Extract patch from nifti images.

    CAPS_DIRECTORY is the CAPS folder where nifti images are stored and tensor will be saved.

    MODALITY [t1-linear|pet-linear|custom] is the clinica pipeline name used for image preprocessing.
    """
    parameters = get_parameters_dict(
        modality_bids,
        "patch",
        save_features,
        extract_json,
        use_uncropped_image,
        custom_suffix,
        tracer,
        suvr_reference_region,
    )
    parameters["patch_size"] = patch_size
    parameters["stride_size"] = stride_size

    DeepLearningPrepareData(
        caps_directory=caps_directory,
        tsv_file=subjects_sessions_tsv,
        n_proc=n_proc,
        parameters=parameters,
        from_bids=bids_directory,
    )


@click.command(name="slice", no_args_is_help=True)
@cli_param.argument.bids_directory
@cli_param.argument.caps_directory
@cli_param.argument.modality_bids
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
        "rgb: Save the slice in three identical channels, "
        "single: Save the slice in a single channel."
    ),
)
@click.option(
    "-ds",
    "--discarded_slices",
    type=int,
    default=(0, 0),
    multiple=2,
    help="""Number of slices discarded from respectively the beginning and
        the end of the MRI volume.  If only one argument is given, it will be
        used for both sides.""",
)
@cli_param.option.tracer
@cli_param.option.suvr_reference_region
@cli_param.option.custom_suffix
def slice_bids_cli(
    bids_directory: Path,
    caps_directory: Path,
    modality_bids: str,
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
    """Extract slice from nifti images.

    CAPS_DIRECTORY is the CAPS folder where nifti images are stored and tensor will be saved.

    MODALITY [t1-linear|pet-linear|custom] is the clinica pipeline name used for image preprocessing.
    """
    parameters = get_parameters_dict(
        modality_bids,
        "slice",
        save_features,
        extract_json,
        use_uncropped_image,
        custom_suffix,
        tracer,
        suvr_reference_region,
    )
    parameters["slice_direction"] = slice_direction
    parameters["slice_mode"] = slice_mode
    parameters["discarded_slices"] = discarded_slices

    DeepLearningPrepareData(
        caps_directory=caps_directory,
        tsv_file=subjects_sessions_tsv,
        n_proc=n_proc,
        parameters=parameters,
        from_bids=bids_directory,
    )


@click.command(name="roi", no_args_is_help=True)
@cli_param.argument.bids_directory
@cli_param.argument.caps_directory
@cli_param.argument.modality_bids
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
    help="Disable cropping option so the output tensors "
    "have the same size than the whole image.",
)
@click.option(
    "--roi_custom_template",
    "-ct",
    type=str,
    default="",
    help="""Template name if MODALITY is `custom`.
        Name of the template used for registration during the preprocessing procedure.""",
)
@click.option(
    "--roi_custom_mask_pattern",
    "-cmp",
    type=str,
    default="",
    help="""Mask pattern if MODALITY is `custom`.
            If given will select only the masks containing the string given.
            The mask with the shortest name is taken.""",
)
@cli_param.option.tracer
@cli_param.option.suvr_reference_region
@cli_param.option.custom_suffix
def roi_bids_cli(
    bids_directory: Path,
    caps_directory: Path,
    modality_bids: str,
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
    """Extract roi from nifti images.

    CAPS_DIRECTORY is the CAPS folder where nifti images are stored and tensor will be saved.

    MODALITY [t1-linear|pet-linear|custom] is the clinica pipeline name used for image preprocessing.
    """
    parameters = get_parameters_dict(
        modality_bids,
        "roi",
        save_features,
        extract_json,
        use_uncropped_image,
        custom_suffix,
        tracer,
        suvr_reference_region,
    )
    parameters["roi_list"] = roi_list
    parameters["uncropped_roi"] = roi_uncrop_output
    parameters["roi_custom_template"] = roi_custom_template
    parameters["roi_custom_mask_pattern"] = roi_custom_mask_pattern

    DeepLearningPrepareData(
        caps_directory=caps_directory,
        tsv_file=subjects_sessions_tsv,
        n_proc=n_proc,
        parameters=parameters,
        from_bids=bids_directory,
    )


class RegistrationOrderGroup(click.Group):
    """CLI group which lists commands by order or registration."""

    def list_commands(self, ctx):
        return self.commands.keys()


@click.group(
    cls=RegistrationOrderGroup, name="prepare-data-from-bids", no_args_is_help=True
)
def cli() -> None:
    """Extract Pytorch tensors from nifti images."""
    pass


cli.add_command(image_bids_cli)
cli.add_command(slice_bids_cli)
cli.add_command(patch_bids_cli)
cli.add_command(roi_bids_cli)


if __name__ == "__main__":
    cli()
