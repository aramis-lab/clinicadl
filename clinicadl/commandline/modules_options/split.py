import click

from clinicadl.config.config_utils import get_default_from_config_class as get_default
from clinicadl.config.config_utils import get_type_from_config_class as get_type
from clinicadl.splitter.splitter.kfold import KFoldConfig
from clinicadl.splitter.splitter.single_split import SingleSplitConfig

# Cross Validation
n_splits = click.option(
    "--n_splits",
    type=get_type("n_splits", KFoldConfig),
    default=get_default("n_splits", KFoldConfig),
    help="If a value is given for k will load data of a k-fold CV. "
    "Default value (0) will load a single split.",
    show_default=True,
)
split = click.option(
    "--split",
    "-s",
    type=int,  # get_type("split", config.ValidationConfig),
    default=get_default("split", SingleSplitConfig),
    multiple=True,
    help="Train the list of given splits. By default, all the splits are trained.",
    show_default=True,
)
