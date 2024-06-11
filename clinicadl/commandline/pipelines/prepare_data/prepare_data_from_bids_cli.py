import click

from clinicadl.caps_dataset.caps_dataset_config import CapsDatasetConfig
from clinicadl.commandline import arguments
from clinicadl.commandline.modules_options import (
    data,
    dataloader,
    modality,
    preprocessing,
)
from clinicadl.prepare_data.prepare_data import DeepLearningPrepareData
from clinicadl.utils.enum import ExtractionMethod


@click.command(name="image", no_args_is_help=True)
@arguments.bids_directory
@arguments.caps_directory
@arguments.modality_bids
@dataloader.n_proc
@preprocessing.preprocessing_json
@preprocessing.use_uncropped_image
@modality.tracer
@modality.suvr_reference_region
@modality.custom_suffix
@data.participants_tsv
def image_bids_cli(kwargs):
    image_config = CapsDatasetConfig.from_preprocessing_and_extraction_method(
        extraction=ExtractionMethod.IMAGE,
        preprocessing=kwargs["preprocessing"],
        **kwargs,
    )

    DeepLearningPrepareData(image_config, from_bids=kwargs["bids_directory"])


@click.command(name="patch", no_args_is_help=True)
@arguments.bids_directory
@arguments.caps_directory
@arguments.modality_bids
@dataloader.n_proc
@preprocessing.save_features
@data.participants_tsv
@preprocessing.preprocessing_json
@preprocessing.use_uncropped_image
@preprocessing.patch_size
@preprocessing.stride_size
@modality.tracer
@modality.suvr_reference_region
@modality.custom_suffix
def patch_bids_cli(kwargs):
    """Extract patch from nifti images.
    CAPS_DIRECTORY is the CAPS folder where nifti images are stored and tensor will be saved.
    MODALITY [t1-linear|pet-linear|custom] is the clinica pipeline name used for image preprocessing.
    """
    patch_config = CapsDatasetConfig.from_preprocessing_and_extraction_method(
        extraction=ExtractionMethod.PATCH,
        preprocessing=kwargs["preprocessing"],
        **kwargs,
    )

    DeepLearningPrepareData(patch_config, from_bids=kwargs["bids_directory"])


@click.command(name="slice", no_args_is_help=True)
@arguments.bids_directory
@arguments.caps_directory
@arguments.modality_bids
@dataloader.n_proc
@preprocessing.save_features
@data.participants_tsv
@preprocessing.preprocessing_json
@preprocessing.use_uncropped_image
@preprocessing.slice_direction
@preprocessing.slice_mode
@preprocessing.discarded_slices
@modality.tracer
@modality.suvr_reference_region
@modality.custom_suffix
def slice_bids_cli(kwargs):
    """Extract slice from nifti images.
    CAPS_DIRECTORY is the CAPS folder where nifti images are stored and tensor will be saved.
    MODALITY [t1-linear|pet-linear|custom] is the clinica pipeline name used for image preprocessing.
    """

    slice_config = CapsDatasetConfig.from_preprocessing_and_extraction_method(
        extraction=ExtractionMethod.SLICE,
        preprocessing=kwargs["preprocessing"],
        **kwargs,
    )

    DeepLearningPrepareData(slice_config, from_bids=kwargs["bids_directory"])


@click.command(name="roi", no_args_is_help=True)
@arguments.bids_directory
@arguments.caps_directory
@arguments.modality_bids
@dataloader.n_proc
@preprocessing.save_features
@data.participants_tsv
@preprocessing.preprocessing_json
@preprocessing.use_uncropped_image
@preprocessing.roi_custom_mask_pattern
@preprocessing.roi_custom_template
@preprocessing.roi_list
@preprocessing.roi_uncrop_output
@modality.tracer
@modality.suvr_reference_region
@modality.custom_suffix
def roi_bids_cli(kwargs):
    """Extract roi from nifti images.

    CAPS_DIRECTORY is the CAPS folder where nifti images are stored and tensor will be saved.

    MODALITY [t1-linear|pet-linear|custom] is the clinica pipeline name used for image preprocessing.
    """
    roi_config = CapsDatasetConfig.from_preprocessing_and_extraction_method(
        extraction=ExtractionMethod.ROI,
        preprocessing=kwargs["preprocessing"],
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
