import click

import clinicadl.train.trainer.training_config as config
from clinicadl.config import config
from clinicadl.utils.config_utils import get_default_from_config_class as get_default
from clinicadl.utils.config_utils import get_type_from_config_class as get_type

# Model
multi_network = click.option(
    "--multi_network/--single_network",
    default=get_default("multi_network", config.ModelConfig),
    help="If provided uses a multi-network framework.",
    show_default=True,
)
dropout = click.option(
    "--dropout",
    type=get_type("dropout", config.ModelConfig),
    default=get_default("dropout", config.ModelConfig),
    help="Rate value applied to dropout layers in a CNN architecture.",
    show_default=True,
)
