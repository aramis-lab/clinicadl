from pathlib import Path
from typing import get_args

import click

from clinicadl.utils.enum import (
    DTIMeasure,
    DTISpace,
    Preprocessing,
    SliceDirection,
    SliceMode,
    SUVRReferenceRegions,
    Tracer,
)
from clinicadl.utils.preprocessing.preprocessing_config import PreprocessingSliceConfig

config = PreprocessingSliceConfig.model_fields

slice_direction = click.option(
    "-sd",
    "--slice_direction",
    type=click.Choice(SliceDirection),
    default=config["slice_direction_cls"].default.value,
    show_default=True,
    help="Slice direction. 0: Sagittal plane, 1: Coronal plane, 2: Axial plane.",
)
slice_method = click.option(
    "-sm",
    "--slice_mode",
    type=click.Choice(SliceMode),
    default=config["slice_mode_cls"].default.value,
    show_default=True,
    help=(
        "rgb: Save the slice in three identical channels, "
        "single: Save the slice in a single channel."
    ),
)
discarded_slice = click.option(
    "-ds",
    "--discarded_slices",
    type=int,  # get_args(config["discarded_slices"].annotation)[0],
    default=config["discarded_slices"].default,
    multiple=2,
    help="""Number of slices discarded from respectively the beginning and
        the end of the MRI volume.  If only one argument is given, it will be
        used for both sides.""",
)
