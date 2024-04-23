import click

from clinicadl.generate.generate_config import GenerateTrivialConfig

config_trivial = GenerateTrivialConfig.model_fields

mask_path = click.option(
    "--mask_path",
    type=config_trivial["mask_path"].annotation,
    default=config_trivial["mask_path"].default,
    help="Path to the extracted masks to generate the two labels. "
    "Default will try to download masks and store them at '~/.cache/clinicadl'.",
)
atrophy_percent = click.option(
    "--atrophy_percent",
    type=config_trivial["mask_path"].annotation,
    default=config_trivial["mask_path"].default,
    help="Percentage of atrophy applied.",
)
