import click

from clinicadl.utils.caps_dataset.data_config import DataConfig
from clinicadl.utils.config_utils import get_default_from_config_class as get_default
from clinicadl.utils.config_utils import get_type_from_config_class as get_type
from clinicadl.utils.enum import (
    Preprocessing,
    SUVRReferenceRegions,
    Tracer,
)

caps_directory = click.argument(
    "caps_directory",
    type=get_type("caps_directory", DataConfig),
)
preprocessing = click.argument(
    "preprocessing",
    type=click.Choice(Preprocessing),
)
