import click

from clinicadl.config.config.cross_validation import CrossValidationConfig
from clinicadl.config.config_utils import get_default_from_config_class as get_default
from clinicadl.config.config_utils import get_type_from_config_class as get_type

# Cross Validation
n_splits = click.option(
    "--n_splits",
    type=get_type("n_splits", CrossValidationConfig),
    default=get_default("n_splits", CrossValidationConfig),
    help="If a value is given for k will load data of a k-fold CV. "
    "Default value (0) will load a single split.",
    show_default=True,
)
split = click.option(
    "--split",
    "-s",
    type=int,  # get_type("split", config.CrossValidationConfig),
    default=get_default("split", CrossValidationConfig),
    multiple=True,
    help="Train the list of given splits. By default, all the splits are trained.",
    show_default=True,
)
