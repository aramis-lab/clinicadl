from typing import get_args

import click

from clinicadl.generate.generate_config import GenerateTrivialConfig

config_trivial = GenerateTrivialConfig.model_fields

mask_path = click.option(
    "--mask_path",
    type=get_args(config_trivial["mask_path"].annotation)[0],
    default=config_trivial["mask_path"].default,
    help="Path to the extracted masks to generate the two labels. "
    "Default will try to download masks and store them at '~/.cache/clinicadl'.",
    show_default=True,
)
atrophy_percent = click.option(
    "--atrophy_percent",
    type=config_trivial["atrophy_percent"].annotation,
    default=config_trivial["atrophy_percent"].default,
    help="Percentage of atrophy applied.",
    show_default=True,
)
