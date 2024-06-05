from pathlib import Path
from typing import Optional

import click

from clinicadl.commandline import arguments
from clinicadl.commandline.modules_options import (
    data,
    dataloader,
    modality,
    preprocessing,
)
from clinicadl.prepare_data.prepare_data import DeepLearningPrepareData


@click.command(name="image", no_args_is_help=True)
@arguments.bids_directory
@arguments.caps_directory
@arguments.modality_bids
@dataloader.n_proc
@preprocessing.extract_json
@preprocessing.use_uncropped_image
@modality.tracer
@modality.suvr_reference_region
@modality.custom_suffix
@data.participants_tsv
def image_bids_cli(
    bids_directory: Path,
    caps_directory: Path,
    modality_bids: str,
    n_proc: int,
    participants_tsv: Optional[Path] = None,
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
        tsv_file=participants_tsv,
        n_proc=n_proc,
        parameters=parameters,
        from_bids=bids_directory,
    )


@click.command(name="patch", no_args_is_help=True)
@arguments.bids_directory
@arguments.caps_directory
@arguments.modality_bids
@dataloader.n_proc
@preprocessing.save_features
@data.participants_tsv
@preprocessing.extract_json
@preprocessing.use_uncropped_image
@preprocessing.patch_size
@preprocessing.stride_size
@modality.tracer
@modality.suvr_reference_region
@modality.custom_suffix
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
@arguments.bids_directory
@arguments.caps_directory
@arguments.modality_bids
@dataloader.n_proc
@preprocessing.save_features
@data.participants_tsv
@preprocessing.extract_json
@preprocessing.use_uncropped_image
@preprocessing.slice_direction
@preprocessing.slice_mode
@preprocessing.discarded_slices
@modality.tracer
@modality.suvr_reference_region
@modality.custom_suffix
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
@arguments.bids_directory
@arguments.caps_directory
@arguments.modality_bids
@dataloader.n_proc
@preprocessing.save_features
@data.participants_tsv
@preprocessing.extract_json
@preprocessing.use_uncropped_image
@preprocessing.roi_custom_mask_pattern
@preprocessing.roi_custom_template
@preprocessing.roi_list
@preprocessing.roi_uncrop_output
@modality.tracer
@modality.suvr_reference_region
@modality.custom_suffix
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
