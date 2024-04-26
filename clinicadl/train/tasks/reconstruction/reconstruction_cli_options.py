from typing import get_args

import click

from clinicadl.utils import cli_param

from .reconstruction_config import ReconstructionConfig

reconstruction_config = ReconstructionConfig.model_fields

architecture = cli_param.option_group.model_group.option(
    "-a",
    "--architecture",
    type=reconstruction_config["architecture"].annotation,
    default=reconstruction_config["architecture"].default,
    help="Architecture of the chosen model to train. A set of model is available in ClinicaDL, default architecture depends on the NETWORK_TASK (see the documentation for more information).",
)
selection_metrics = cli_param.option_group.task_group.option(
    "--selection_metrics",
    "-sm",
    multiple=True,
    type=get_args(reconstruction_config["selection_metrics"].annotation)[0],
    default=reconstruction_config["selection_metrics"].default,
    help="""Allow to save a list of models based on their selection metric. Default will
    only save the best model selected on loss.""",
    show_default=True,
)
loss = cli_param.option_group.task_group.option(
    "--loss",
    "-l",
    type=click.Choice(ReconstructionConfig.get_compatible_losses()),
    default=reconstruction_config["loss"].default,
    help="Loss used by the network to optimize its training task.",
    show_default=True,
)
