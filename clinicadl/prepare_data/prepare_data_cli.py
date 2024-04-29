from pathlib import Path
from typing import Optional

import click

from clinicadl.prepare_data import prepare_data_param
from clinicadl.prepare_data.prepare_data_config import (
    PrepareDataImageConfig,
    PrepareDataPatchConfig,
    PrepareDataROIConfig,
    PrepareDataSliceConfig,
)
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

from .prepare_data import DeepLearningPrepareData


@click.command(name="image", no_args_is_help=True)
@prepare_data_param.argument.caps_directory
@prepare_data_param.argument.preprocessing
@prepare_data_param.option.n_proc
@prepare_data_param.option.tsv_file
@prepare_data_param.option.extract_json
@prepare_data_param.option.use_uncropped_image
@prepare_data_param.option.tracer
@prepare_data_param.option.suvr_reference_region
@prepare_data_param.option.custom_suffix
@prepare_data_param.option.dti_measure
@prepare_data_param.option.dti_space
def image_cli(
    caps_directory: Path,
    preprocessing: Preprocessing,
    **kwargs,
):
    """Extract image from nifti images.

    CAPS_DIRECTORY is the CAPS folder where nifti images are stored and tensor will be saved.

    MODALITY [t1-linear|pet-linear|custom] is the clinica pipeline name used for image preprocessing.
    """
    image_config = PrepareDataImageConfig(
        caps_directory=caps_directory,
        preprocessing_cls=preprocessing,
        tracer_cls=kwargs["tracer"],
        suvr_reference_region_cls=kwargs["suvr_reference_region"],
        dti_measure_cls=kwargs["dti_measure"],
        dti_space_cls=kwargs["dti_space"],
        save_features=True,
        **kwargs,
    )

    DeepLearningPrepareData(image_config)


@click.command(name="patch", no_args_is_help=True)
@prepare_data_param.argument.caps_directory
@prepare_data_param.argument.preprocessing
@prepare_data_param.option.n_proc
@prepare_data_param.option.save_features
@prepare_data_param.option.tsv_file
@prepare_data_param.option.extract_json
@prepare_data_param.option.use_uncropped_image
@prepare_data_param.option.tracer
@prepare_data_param.option.suvr_reference_region
@prepare_data_param.option.custom_suffix
@prepare_data_param.option.dti_measure
@prepare_data_param.option.dti_space
@prepare_data_param.option_patch.patch_size
@prepare_data_param.option_patch.stride_size
def patch_cli(caps_directory: Path, preprocessing: Preprocessing, **kwargs):
    """Extract patch from nifti images.

    CAPS_DIRECTORY is the CAPS folder where nifti images are stored and tensor will be saved.

    MODALITY [t1-linear|pet-linear|custom] is the clinica pipeline name used for image preprocessing.
    """

    patch_config = PrepareDataPatchConfig(
        caps_directory=caps_directory,
        preprocessing_cls=preprocessing,
        tracer_cls=kwargs["tracer"],
        suvr_reference_region_cls=kwargs["suvr_reference_region"],
        dti_measure_cls=kwargs["dti_measure"],
        dti_space_cls=kwargs["dti_space"],
        **kwargs,
    )

    DeepLearningPrepareData(patch_config)


@click.command(name="slice", no_args_is_help=True)
@prepare_data_param.argument.caps_directory
@prepare_data_param.argument.preprocessing
@prepare_data_param.option.n_proc
@prepare_data_param.option.save_features
@prepare_data_param.option.tsv_file
@prepare_data_param.option.extract_json
@prepare_data_param.option.use_uncropped_image
@prepare_data_param.option.tracer
@prepare_data_param.option.suvr_reference_region
@prepare_data_param.option.custom_suffix
@prepare_data_param.option.dti_measure
@prepare_data_param.option.dti_space
@prepare_data_param.option_slice.slice_method
@prepare_data_param.option_slice.slice_direction
@prepare_data_param.option_slice.discarded_slice
def slice_cli(caps_directory: Path, preprocessing: Preprocessing, **kwargs):
    """Extract slice from nifti images.

    CAPS_DIRECTORY is the CAPS folder where nifti images are stored and tensor will be saved.

    MODALITY [t1-linear|pet-linear|custom] is the clinica pipeline name used for image preprocessing.
    """

    slice_config = PrepareDataSliceConfig(
        caps_directory=caps_directory,
        preprocessing_cls=preprocessing,
        tracer_cls=kwargs["tracer"],
        suvr_reference_region_cls=kwargs["suvr_reference_region"],
        dti_measure_cls=kwargs["dti_measure"],
        dti_space_cls=kwargs["dti_space"],
        slice_direction_cls=kwargs["slice_direction"],
        slice_mode_cls=kwargs["slice_mode"],
        **kwargs,
    )

    DeepLearningPrepareData(slice_config)


@click.command(name="roi", no_args_is_help=True)
@prepare_data_param.argument.caps_directory
@prepare_data_param.argument.preprocessing
@prepare_data_param.option.n_proc
@prepare_data_param.option.save_features
@prepare_data_param.option.tsv_file
@prepare_data_param.option.extract_json
@prepare_data_param.option.use_uncropped_image
@prepare_data_param.option.tracer
@prepare_data_param.option.suvr_reference_region
@prepare_data_param.option.custom_suffix
@prepare_data_param.option.dti_measure
@prepare_data_param.option.dti_space
@prepare_data_param.option_roi.roi_list
@prepare_data_param.option_roi.roi_uncrop_output
@prepare_data_param.option_roi.roi_custom_template
@prepare_data_param.option_roi.roi_custom_mask_pattern
def roi_cli(caps_directory: Path, preprocessing: Preprocessing, **kwargs):
    """Extract roi from nifti images.

    CAPS_DIRECTORY is the CAPS folder where nifti images are stored and tensor will be saved.

    MODALITY [t1-linear|pet-linear|custom] is the clinica pipeline name used for image preprocessing.
    """

    roi_config = PrepareDataROIConfig(
        caps_directory=caps_directory,
        preprocessing_cls=preprocessing,
        tracer_cls=kwargs["tracer"],
        suvr_reference_region_cls=kwargs["suvr_reference_region"],
        dti_measure_cls=kwargs["dti_measure"],
        dti_space_cls=kwargs["dti_space"],
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
