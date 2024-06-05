from pathlib import Path
from typing import Optional

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
from clinicadl.utils.enum import (
    BIDSModality,
    DTIMeasure,
    DTISpace,
    ExtractionMethod,
    Pathology,
    Preprocessing,
    SUVRReferenceRegions,
    Tracer,
)


@click.command(name="image", no_args_is_help=True)
@arguments.caps_directory
@arguments.preprocessing
@dataloader.n_proc
@data.participants_tsv
@preprocessing.extract_json
@preprocessing.use_uncropped_image
@modality.tracer
@modality.suvr_reference_region
@modality.custom_suffix
@modality.dti_measure
@modality.dti_space
def image_cli(**kwargs):
    """Extract image from nifti images.

    CAPS_DIRECTORY is the CAPS folder where nifti images are stored and tensor will be saved.

    PREPROCESSING [t1-linear|pet-linear|custom] is the clinica pipeline name used for image preprocessing.
    """
    kwargs["save_features"] = True
    image_config = CapsDatasetConfig.from_preprocessing_and_extraction_method(
        extraction=ExtractionMethod.IMAGE,
        preprocessing_type=kwargs["preprocessing"],
        **kwargs,
    )

    DeepLearningPrepareData(image_config)


@click.command(name="patch", no_args_is_help=True)
@arguments.caps_directory
@arguments.preprocessing
@dataloader.n_proc
@preprocessing.save_features
@data.participants_tsv
@preprocessing.extract_json
@preprocessing.use_uncropped_image
@modality.tracer
@modality.suvr_reference_region
@modality.custom_suffix
@modality.dti_measure
@modality.dti_space
@preprocessing.patch_size
@preprocessing.stride_size
def patch_cli(**kwargs):
    """Extract patch from nifti images.

    CAPS_DIRECTORY is the CAPS folder where nifti images are stored and tensor will be saved.

    PREPROCESSING [t1-linear|pet-linear|custom] is the clinica pipeline name used for image preprocessing.
    """

    patch_config = CapsDatasetConfig.from_preprocessing_and_extraction_method(
        extraction=ExtractionMethod.PATCH,
        preprocessing_type=kwargs["preprocessing"],
        **kwargs,
    )

    DeepLearningPrepareData(patch_config)


@click.command(name="slice", no_args_is_help=True)
@arguments.caps_directory
@arguments.preprocessing
@dataloader.n_proc
@preprocessing.save_features
@data.participants_tsv
@preprocessing.extract_json
@preprocessing.use_uncropped_image
@modality.tracer
@modality.suvr_reference_region
@modality.custom_suffix
@modality.dti_measure
@modality.dti_space
@preprocessing.slice_mode
@preprocessing.slice_direction
@preprocessing.discarded_slices
def slice_cli(**kwargs):
    """Extract slice from nifti images.

    CAPS_DIRECTORY is the CAPS folder where nifti images are stored and tensor will be saved.

    PREPROCESSING [t1-linear|pet-linear|custom] is the clinica pipeline name used for image preprocessing.
    """
    slice_config = CapsDatasetConfig.from_preprocessing_and_extraction_method(
        extraction=ExtractionMethod.SLICE,
        preprocessing_type=kwargs["preprocessing"],
        **kwargs,
    )

    DeepLearningPrepareData(slice_config)


@click.command(name="roi", no_args_is_help=True)
@arguments.caps_directory
@arguments.preprocessing
@dataloader.n_proc
@preprocessing.save_features
@data.participants_tsv
@preprocessing.extract_json
@preprocessing.use_uncropped_image
@modality.tracer
@modality.suvr_reference_region
@modality.custom_suffix
@modality.dti_measure
@modality.dti_space
@preprocessing.roi_list
@preprocessing.roi_uncrop_output
@preprocessing.roi_custom_template
@preprocessing.roi_custom_mask_pattern
def roi_cli(**kwargs):
    """Extract roi from nifti images.

    CAPS_DIRECTORY is the CAPS folder where nifti images are stored and tensor will be saved.

    PREPROCESSING [t1-linear|pet-linear|custom] is the clinica pipeline name used for image preprocessing.
    """

    roi_config = CapsDatasetConfig.from_preprocessing_and_extraction_method(
        extraction=ExtractionMethod.ROI,
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
