import click

from clinicadl.caps_dataset.caps_dataset_config import CapsDatasetConfig
from clinicadl.commandline import arguments
from clinicadl.commandline.modules_options import (
    data,
    dataloader,
    extraction,
    preprocessing,
)
from clinicadl.prepare_data.prepare_data import DeepLearningPrepareData
from clinicadl.utils.enum import ExtractionMethod


@click.command(name="image", no_args_is_help=True)
@arguments.bids_directory
@arguments.caps_directory
@arguments.modality_bids
@dataloader.n_proc
@extraction.extract_json
@extraction.use_uncropped_image
@preprocessing.tracer
@preprocessing.suvr_reference_region
@preprocessing.custom_suffix
@data.participants_tsv
def image_bids_cli(kwargs):
    image_config = CapsDatasetConfig.from_preprocessing_and_extraction_method(
        extraction=ExtractionMethod.IMAGE,
        preprocessing_type=kwargs["preprocessing"],
        **kwargs,
    )

    DeepLearningPrepareData(image_config, from_bids=kwargs["bids_directory"])


@click.command(name="patch", no_args_is_help=True)
@arguments.bids_directory
@arguments.caps_directory
@arguments.modality_bids
@dataloader.n_proc
@extraction.save_features
@data.participants_tsv
@extraction.extract_json
@extraction.use_uncropped_image
@extraction.patch_size
@extraction.stride_size
@preprocessing.tracer
@preprocessing.suvr_reference_region
@preprocessing.custom_suffix
def patch_bids_cli(kwargs):
    """Extract patch from nifti images.
    CAPS_DIRECTORY is the CAPS folder where nifti images are stored and tensor will be saved.
    MODALITY [t1-linear|pet-linear|custom] is the clinica pipeline name used for image preprocessing.
    """
    patch_config = CapsDatasetConfig.from_preprocessing_and_extraction_method(
        extraction=ExtractionMethod.PATCH,
        preprocessing_type=kwargs["preprocessing"],
        **kwargs,
    )

    DeepLearningPrepareData(patch_config, from_bids=kwargs["bids_directory"])


@click.command(name="slice", no_args_is_help=True)
@arguments.bids_directory
@arguments.caps_directory
@arguments.modality_bids
@dataloader.n_proc
@extraction.save_features
@data.participants_tsv
@extraction.extract_json
@extraction.use_uncropped_image
@extraction.slice_direction
@extraction.slice_mode
@extraction.discarded_slices
@preprocessing.tracer
@preprocessing.suvr_reference_region
@preprocessing.custom_suffix
def slice_bids_cli(kwargs):
    """Extract slice from nifti images.
    CAPS_DIRECTORY is the CAPS folder where nifti images are stored and tensor will be saved.
    MODALITY [t1-linear|pet-linear|custom] is the clinica pipeline name used for image preprocessing.
    """

    slice_config = CapsDatasetConfig.from_preprocessing_and_extraction_method(
        extraction=ExtractionMethod.SLICE,
        preprocessing_type=kwargs["preprocessing"],
        **kwargs,
    )

    DeepLearningPrepareData(slice_config, from_bids=kwargs["bids_directory"])


@click.command(name="roi", no_args_is_help=True)
@arguments.bids_directory
@arguments.caps_directory
@arguments.modality_bids
@dataloader.n_proc
@extraction.save_features
@data.participants_tsv
@extraction.extract_json
@extraction.use_uncropped_image
@extraction.roi_custom_mask_pattern
@extraction.roi_custom_template
@extraction.roi_list
@extraction.roi_uncrop_output
@preprocessing.tracer
@preprocessing.suvr_reference_region
@preprocessing.custom_suffix
def roi_bids_cli(kwargs):
    """Extract roi from nifti images.

    CAPS_DIRECTORY is the CAPS folder where nifti images are stored and tensor will be saved.

    MODALITY [t1-linear|pet-linear|custom] is the clinica pipeline name used for image preprocessing.
    """
    roi_config = CapsDatasetConfig.from_preprocessing_and_extraction_method(
        extraction=ExtractionMethod.ROI,
        preprocessing_type=kwargs["preprocessing"],
        **kwargs,
    )

    DeepLearningPrepareData(roi_config, from_bids=kwargs["bids_directory"])


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
