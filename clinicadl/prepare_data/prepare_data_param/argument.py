from pathlib import Path

import click

from clinicadl.prepare_data.prepare_data_config import PrepareDataConfig
from clinicadl.utils.enum import (
    Preprocessing,
    SUVRReferenceRegions,
    Tracer,
)

config = PrepareDataConfig.model_fields

caps_directory = click.argument(
    "caps_directory", type=config["caps_directory"].annotation
)
preprocessing = click.argument(
    "preprocessing",
    type=click.Choice(Preprocessing),
    help="Preprocessing used to generate synthetic data.",
)
