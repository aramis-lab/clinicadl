import click

from clinicadl.config.config_utils import get_default_from_config_class as get_default
from clinicadl.config.config_utils import get_type_from_config_class as get_type
from clinicadl.optimizer.optimizer import OptimizerConfig

# Optimizer
learning_rate = click.option(
    "--learning_rate",
    "-lr",
    type=get_type("learning_rate", OptimizerConfig),
    default=get_default("learning_rate", OptimizerConfig),
    help="Learning rate of the optimization.",
    show_default=True,
)
optimizer = click.option(
    "--optimizer",
    type=get_type("optimizer", OptimizerConfig),
    default=get_default("optimizer", OptimizerConfig),
    help="Optimizer used to train the network.",
    show_default=True,
)
weight_decay = click.option(
    "--weight_decay",
    "-wd",
    type=get_type("weight_decay", OptimizerConfig),
    default=get_default("weight_decay", OptimizerConfig),
    help="Weight decay value used in optimization.",
    show_default=True,
)
