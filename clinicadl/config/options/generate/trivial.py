from typing import get_args

import click

from clinicadl.generate.generate_config import GenerateTrivialConfig
from clinicadl.utils.config_utils import get_default_from_config_class as get_default
from clinicadl.utils.config_utils import get_type_from_config_class as get_type

mask_path = click.option(
    "--mask_path",
    type=get_type("mask_path", GenerateTrivialConfig),
    default=get_default("mask_path", GenerateTrivialConfig),
    help="Path to the extracted masks to generate the two labels. "
    "Default will try to download masks and store them at '~/.cache/clinicadl'.",
    show_default=True,
)
atrophy_percent = click.option(
    "--atrophy_percent",
    type=get_type("atrophy_percent", GenerateTrivialConfig),
    default=get_default("atrophy_percent", GenerateTrivialConfig),
    help="Percentage of atrophy applied.",
    show_default=True,
)
