from pathlib import Path
from typing import get_args

import click

from clinicadl.prepare_data.prepare_data_config import PrepareDataPatchConfig
from clinicadl.utils.enum import (
    DTIMeasure,
    DTISpace,
    Preprocessing,
    SUVRReferenceRegions,
    Tracer,
)

config = PrepareDataPatchConfig.model_fields

patch_size = click.option(
    "-ps",
    "--patch_size",
    default=50,
    show_default=True,
    help="Patch size.",
)
stride_size = click.option(
    "-ss",
    "--stride_size",
    default=50,
    show_default=True,
    help="Stride size.",
)
