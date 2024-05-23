import click

from clinicadl.utils import cli_param
from clinicadl.utils.config_utils import get_default_from_config_class as get_default
from clinicadl.utils.config_utils import get_type_from_config_class as get_type

from .reconstruction_config import ModelConfig, ValidationConfig

# Model
architecture = cli_param.option_group.model_group.option(
    "-a",
    "--architecture",
    type=get_type("architecture", ModelConfig),
    default=get_default("architecture", ModelConfig),
    help="Architecture of the chosen model to train. A set of model is available in ClinicaDL, default architecture depends on the NETWORK_TASK (see the documentation for more information).",
)
loss = cli_param.option_group.task_group.option(
    "--loss",
    "-l",
    type=click.Choice(get_type("loss", ModelConfig)),
    default=get_default("loss", ModelConfig),
    help="Loss used by the network to optimize its training task.",
    show_default=True,
)
# Validation
selection_metrics = cli_param.option_group.task_group.option(
    "--selection_metrics",
    "-sm",
    multiple=True,
    type=click.Choice(get_type("selection_metrics", ValidationConfig)),
    default=get_default("selection_metrics", ValidationConfig),
    help="""Allow to save a list of models based on their selection metric. Default will
    only save the best model selected on loss.""",
    show_default=True,
)
