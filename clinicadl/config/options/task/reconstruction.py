import click

from clinicadl.train.tasks.reconstruction.config import ModelConfig, ValidationConfig
from clinicadl.utils.config_utils import get_default_from_config_class as get_default
from clinicadl.utils.config_utils import get_type_from_config_class as get_type

# Model
architecture = click.option(
    "-a",
    "--architecture",
    type=get_type("architecture", ModelConfig),
    default=get_default("architecture", ModelConfig),
    help="Architecture of the chosen model to train. A set of model is available in ClinicaDL, default architecture depends on the NETWORK_TASK (see the documentation for more information).",
)
loss = click.option(
    "--loss",
    "-l",
    type=get_type("loss", ModelConfig),
    default=get_default("loss", ModelConfig),
    help="Loss used by the network to optimize its training task.",
    show_default=True,
)
# Validation
selection_metrics = click.option(
    "--selection_metrics",
    "-sm",
    multiple=True,
    type=get_type("selection_metrics", ValidationConfig),
    default=get_default("selection_metrics", ValidationConfig),
    help="""Allow to save a list of models based on their selection metric. Default will
    only save the best model selected on loss.""",
    show_default=True,
)
