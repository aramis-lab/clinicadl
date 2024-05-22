import click

import clinicadl.train.trainer.training_config as config
from clinicadl.config import config
from clinicadl.utils.config_utils import get_default_from_config_class as get_default
from clinicadl.utils.config_utils import get_type_from_config_class as get_type

# Early Stopping
patience = click.option(
    "--patience",
    type=get_type("patience", config.EarlyStoppingConfig),
    default=get_default("patience", config.EarlyStoppingConfig),
    help="Number of epochs for early stopping patience.",
    show_default=True,
)
tolerance = click.option(
    "--tolerance",
    type=get_type("tolerance", config.EarlyStoppingConfig),
    default=get_default("tolerance", config.EarlyStoppingConfig),
    help="Value for early stopping tolerance.",
    show_default=True,
)
