from pathlib import Path
from typing import Optional

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
@arguments.caps_directory
@arguments.preprocessing
@dataloader.n_proc
@data.participants_tsv
@data.preprocessing_json
@extraction.use_uncropped_image
@preprocessing.tracer
@preprocessing.suvr_reference_region
@preprocessing.custom_suffix
@preprocessing.dti_measure
@preprocessing.dti_space
def image_cli(**kwargs):
    """Extract image from nifti images.

    CAPS_DIRECTORY is the CAPS folder where nifti images are stored and tensor will be saved.

    PREPROCESSING [t1-linear|pet-linear|custom] is the clinica pipeline name used for image preprocessing.
    """
    kwargs["save_features"] = True
    image_config = CapsDatasetConfig.from_preprocessing_and_extraction_method(
        extraction_type=ExtractionMethod.IMAGE,
        preprocessing_type=kwargs["preprocessing"],
        **kwargs,
    )

    DeepLearningPrepareData(image_config)


@click.command(name="patch", no_args_is_help=True)
@arguments.caps_directory
@arguments.preprocessing
@dataloader.n_proc
@extraction.save_features
@data.participants_tsv
@data.preprocessing_json
@extraction.use_uncropped_image
@preprocessing.tracer
@preprocessing.suvr_reference_region
@preprocessing.custom_suffix
@preprocessing.dti_measure
@preprocessing.dti_space
@extraction.patch_size
@extraction.stride_size
def patch_cli(**kwargs):
    """Extract patch from nifti images.

    CAPS_DIRECTORY is the CAPS folder where nifti images are stored and tensor will be saved.

    PREPROCESSING [t1-linear|pet-linear|custom] is the clinica pipeline name used for image preprocessing.
    """

    patch_config = CapsDatasetConfig.from_preprocessing_and_extraction_method(
        extraction_type=ExtractionMethod.PATCH,
        preprocessing_type=kwargs["preprocessing"],
        **kwargs,
    )

    DeepLearningPrepareData(patch_config)


@click.command(name="slice", no_args_is_help=True)
@arguments.caps_directory
@arguments.preprocessing
@dataloader.n_proc
@extraction.save_features
@data.participants_tsv
@data.preprocessing_json
@extraction.use_uncropped_image
@preprocessing.tracer
@preprocessing.suvr_reference_region
@preprocessing.custom_suffix
@preprocessing.dti_measure
@preprocessing.dti_space
@extraction.slice_mode
@extraction.slice_direction
@extraction.discarded_slices
def slice_cli(**kwargs):
    """Extract slice from nifti images.

    CAPS_DIRECTORY is the CAPS folder where nifti images are stored and tensor will be saved.

    PREPROCESSING [t1-linear|pet-linear|custom] is the clinica pipeline name used for image preprocessing.
    """
    slice_config = CapsDatasetConfig.from_preprocessing_and_extraction_method(
        extraction=ExtractionMethod.SLICE,
        preprocessing=kwargs["preprocessing"],
        **kwargs,
    )

    DeepLearningPrepareData(slice_config)


@click.command(name="roi", no_args_is_help=True)
@arguments.caps_directory
@arguments.preprocessing
@dataloader.n_proc
@extraction.save_features
@data.participants_tsv
@data.preprocessing_json
@extraction.use_uncropped_image
@preprocessing.tracer
@preprocessing.suvr_reference_region
@preprocessing.custom_suffix
@preprocessing.dti_measure
@preprocessing.dti_space
@extraction.roi_list
@extraction.roi_uncrop_output
@extraction.roi_custom_template
@extraction.roi_custom_mask_pattern
def roi_cli(**kwargs):
    """Extract roi from nifti images.

    CAPS_DIRECTORY is the CAPS folder where nifti images are stored and tensor will be saved.

    PREPROCESSING [t1-linear|pet-linear|custom] is the clinica pipeline name used for image preprocessing.
    """

    roi_config = CapsDatasetConfig.from_preprocessing_and_extraction_method(
        extraction_type=ExtractionMethod.ROI,
        preprocessing_type=kwargs["preprocessing"],
        **kwargs,
    )

    DeepLearningPrepareData(roi_config)


class RegistrationOrderGroup(click.Group):
    """CLI group which lists commands by order or registration."""

    def list_commands(self, ctx):
        return self.commands.keys()


@click.group(cls=RegistrationOrderGroup, name="prepare-data", no_args_is_help=True)
def cli() -> None:
    """Extract Pytorch tensors from nifti images."""
    pass


cli.add_command(image_cli)
cli.add_command(slice_cli)
cli.add_command(patch_cli)
cli.add_command(roi_cli)


if __name__ == "__main__":
    cli()
