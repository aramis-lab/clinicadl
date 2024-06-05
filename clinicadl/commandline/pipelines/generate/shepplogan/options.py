from typing import get_args

import click

from clinicadl.config.config_utils import get_default_from_config_class as get_default
from clinicadl.config.config_utils import get_type_from_config_class as get_type
from clinicadl.generate.generate_config import GenerateSheppLoganConfig

extract_json = click.option(
    "-ej",
    "--extract_json",
    type=get_type("extract_json", GenerateSheppLoganConfig),
    default=get_default("extract_json", GenerateSheppLoganConfig),
    help="Name of the JSON file created to describe the tensor extraction. "
    "Default will use format extract_{time_stamp}.json",
    show_default=True,
)

image_size = click.option(
    "--image_size",
    help="Size in pixels of the squared images.",
    type=get_type("image_size", GenerateSheppLoganConfig),
    default=get_default("image_size", GenerateSheppLoganConfig),
    show_default=True,
)

cn_subtypes_distribution = click.option(
    "--cn_subtypes_distribution",
    "-csd",
    multiple=3,
    type=get_type("cn_subtypes_distribution", GenerateSheppLoganConfig),
    default=get_default("cn_subtypes_distribution", GenerateSheppLoganConfig),
    help="Probability of each subtype to be drawn in CN label.",
    show_default=True,
)

ad_subtypes_distribution = click.option(
    "--ad_subtypes_distribution",
    "-asd",
    multiple=3,
    type=get_type("ad_subtypes_distribution", GenerateSheppLoganConfig),
    default=get_default("ad_subtypes_distribution", GenerateSheppLoganConfig),
    help="Probability of each subtype to be drawn in AD label.",
    show_default=True,
)

smoothing = click.option(
    "--smoothing/--no-smoothing",
    default=get_type("smoothing", GenerateSheppLoganConfig),
    help="Adds random smoothing to generated data.",
    show_default=True,
)
