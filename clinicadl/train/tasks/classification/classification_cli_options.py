import click

from clinicadl.utils import cli_param
from clinicadl.utils.config_utils import get_default_from_config_class as get_default
from clinicadl.utils.config_utils import get_type_from_config_class as get_type

from .classification_config import ClassificationConfig

classification_config = ClassificationConfig

architecture = cli_param.option_group.model_group.option(
    "-a",
    "--architecture",
    type=get_type("architecture", classification_config),
    default=get_default("architecture", classification_config),
    help="Architecture of the chosen model to train. A set of model is available in ClinicaDL, default architecture depends on the NETWORK_TASK (see the documentation for more information).",
)
label = cli_param.option_group.task_group.option(
    "--label",
    type=get_type("label", classification_config),
    default=get_default("label", classification_config),
    help="Target label used for training.",
    show_default=True,
)
selection_metrics = cli_param.option_group.task_group.option(
    "--selection_metrics",
    "-sm",
    multiple=True,
    type=click.Choice(get_type("selection_metrics", classification_config)),
    default=get_default("selection_metrics", classification_config),
    help="""Allow to save a list of models based on their selection metric. Default will
    only save the best model selected on loss.""",
    show_default=True,
)
threshold = cli_param.option_group.task_group.option(
    "--selection_threshold",
    type=get_type("selection_threshold", classification_config),
    default=get_default("selection_threshold", classification_config),
    help="""Selection threshold for soft-voting. Will only be used if num_networks > 1.""",
    show_default=True,
)
loss = cli_param.option_group.task_group.option(
    "--loss",
    "-l",
    type=click.Choice(get_type("loss", classification_config)),
    default=get_default("loss", classification_config),
    help="Loss used by the network to optimize its training task.",
    show_default=True,
)
