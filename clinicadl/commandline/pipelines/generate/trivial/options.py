from typing import get_args

import click

from clinicadl.config.config_utils import get_default_from_config_class as get_default
from clinicadl.config.config_utils import get_type_from_config_class as get_type
from clinicadl.generate.generate_config import GenerateTrivialConfig

atrophy_percent = click.option(
    "--atrophy_percent",
    type=get_type("atrophy_percent", GenerateTrivialConfig),
    default=get_default("atrophy_percent", GenerateTrivialConfig),
    help="Percentage of atrophy applied.",
    show_default=True,
)
