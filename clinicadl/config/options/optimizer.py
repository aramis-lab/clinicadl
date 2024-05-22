import click

import clinicadl.train.trainer.training_config as config
from clinicadl.config import config
from clinicadl.utils.config_utils import get_default_from_config_class as get_default
from clinicadl.utils.config_utils import get_type_from_config_class as get_type

# Optimizer
learning_rate = click.option(
    "--learning_rate",
    "-lr",
    type=get_type("learning_rate", config.OptimizerConfig),
    default=get_default("learning_rate", config.OptimizerConfig),
    help="Learning rate of the optimization.",
    show_default=True,
)
optimizer = click.option(
    "--optimizer",
    type=get_type("optimizer", config.OptimizerConfig),
    default=get_default("optimizer", config.OptimizerConfig),
    help="Optimizer used to train the network.",
    show_default=True,
)
weight_decay = click.option(
    "--weight_decay",
    "-wd",
    type=get_type("weight_decay", config.OptimizerConfig),
    default=get_default("weight_decay", config.OptimizerConfig),
    help="Weight decay value used in optimization.",
    show_default=True,
)
