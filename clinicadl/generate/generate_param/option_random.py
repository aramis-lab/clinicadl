import click

from clinicadl.generate.generate_config import GenerateRandomConfig

config_random = GenerateRandomConfig.model_fields

mean = click.option(
    "--mean",
    type=config_random["mean"].annotation,
    default=config_random["mean"].default,
    help="Mean value of the gaussian noise added to synthetic images.",
)
sigma = click.option(
    "--sigma",
    type=config_random["sigma"].annotation,
    default=config_random["sigma"].default,
    help="Standard deviation of the gaussian noise added to synthetic images.",
)
