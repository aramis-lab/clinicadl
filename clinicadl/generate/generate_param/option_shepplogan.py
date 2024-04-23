from typing import get_args

import click

from clinicadl.generate.generate_config import GenerateSheppLoganConfig

config_shepplogan = GenerateSheppLoganConfig.model_fields

extract_json = click.option(
    "-ej",
    "--extract_json",
    type=config_shepplogan["extract_json"].annotation,
    default=config_shepplogan["extract_json"].default,
    help="Name of the JSON file created to describe the tensor extraction. "
    "Default will use format extract_{time_stamp}.json",
)

image_size = click.option(
    "--image_size",
    help="Size in pixels of the squared images.",
    type=config_shepplogan["image_size"].annotation,
    default=config_shepplogan["image_size"].default,
)

cn_subtypes_distribution = click.option(
    "--cn_subtypes_distribution",
    "-csd",
    multiple=3,
    type=get_args(config_shepplogan["cn_subtypes_distribution"].annotation)[0],
    default=config_shepplogan["cn_subtypes_distribution"].default,
    help="Probability of each subtype to be drawn in CN label.",
)

ad_subtypes_distribution = click.option(
    "--ad_subtypes_distribution",
    "-asd",
    multiple=3,
    type=get_args(config_shepplogan["ad_subtypes_distribution"].annotation)[0],
    default=config_shepplogan["ad_subtypes_distribution"].default,
    help="Probability of each subtype to be drawn in AD label.",
)

smoothing = click.option(
    "--smoothing/--no-smoothing",
    type=config_shepplogan["smoothing"].annotation,
    default=config_shepplogan["smoothing"].default,
    help="Adds random smoothing to generated data.",
)
