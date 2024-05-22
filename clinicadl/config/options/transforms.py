import click

import clinicadl.train.trainer.training_config as config
from clinicadl.config import config
from clinicadl.utils.config_utils import get_default_from_config_class as get_default
from clinicadl.utils.config_utils import get_type_from_config_class as get_type

# Transform
data_augmentation = click.option(
    "--data_augmentation",
    "-da",
    type=get_type("data_augmentation", config.TransformsConfig),
    default=get_default("data_augmentation", config.TransformsConfig),
    multiple=True,
    help="Randomly applies transforms on the training set.",
    show_default=True,
)
normalize = click.option(
    "--normalize/--unnormalize",
    default=get_default("normalize", config.TransformsConfig),
    help="Disable default MinMaxNormalization.",
    show_default=True,
)
