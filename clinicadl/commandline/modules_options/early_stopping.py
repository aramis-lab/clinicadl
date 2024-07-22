import click

from clinicadl.config.config_utils import get_default_from_config_class as get_default
from clinicadl.config.config_utils import get_type_from_config_class as get_type
from clinicadl.utils.early_stopping.config import EarlyStoppingConfig

# Early Stopping
patience = click.option(
    "--patience",
    type=get_type("patience", EarlyStoppingConfig),
    default=get_default("patience", EarlyStoppingConfig),
    help="Number of epochs for early stopping patience.",
    show_default=True,
)
tolerance = click.option(
    "--tolerance",
    type=get_type("tolerance", EarlyStoppingConfig),
    default=get_default("tolerance", EarlyStoppingConfig),
    help="Value for early stopping tolerance.",
    show_default=True,
)
