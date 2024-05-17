from pathlib import Path
from typing import get_args

import click

from clinicadl.utils.enum import (
    DTIMeasure,
    DTISpace,
    Preprocessing,
    SUVRReferenceRegions,
    Tracer,
)
from clinicadl.utils.preprocessing.preprocessing_config import PreprocessingROIConfig

config = PreprocessingROIConfig.model_fields

roi_list = click.option(
    "--roi_list",
    type=get_args(config["roi_list"].annotation)[0],
    default=config["roi_list"].default,
    multiple=True,
    help="List of regions to be extracted",
)
roi_uncrop_output = click.option(
    "--roi_uncrop_output",
    is_flag=True,
    help="Disable cropping option so the output tensors "
    "have the same size than the whole image.",
)
roi_custom_template = click.option(
    "--roi_custom_template",
    "-ct",
    type=config["roi_custom_template"].annotation,
    default=config["roi_custom_template"].default,
    help="""Template name if MODALITY is `custom`.
        Name of the template used for registration during the preprocessing procedure.""",
)
roi_custom_mask_pattern = click.option(
    "--roi_custom_mask_pattern",
    "-cmp",
    type=config["roi_custom_mask_pattern"].annotation,
    default=config["roi_custom_mask_pattern"].default,
    help="""Mask pattern if MODALITY is `custom`.
            If given will select only the masks containing the string given.
            The mask with the shortest name is taken.""",
)
