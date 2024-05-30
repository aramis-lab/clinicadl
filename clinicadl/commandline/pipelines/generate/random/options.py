import click

from clinicadl.config.config_utils import get_default_from_config_class as get_default
from clinicadl.config.config_utils import get_type_from_config_class as get_type
from clinicadl.generate.generate_config import GenerateRandomConfig

mean = click.option(
    "--mean",
    type=get_type("mean", GenerateRandomConfig),
    default=get_default("mean", GenerateRandomConfig),
    help="Mean value of the gaussian noise added to synthetic images.",
    show_default=True,
)
sigma = click.option(
    "--sigma",
    type=get_type("sigma", GenerateRandomConfig),
    default=get_default("sigma", GenerateRandomConfig),
    help="Standard deviation of the gaussian noise added to synthetic images.",
    show_default=True,
)
