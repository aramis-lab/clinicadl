import click

import clinicadl.train.trainer.training_config as config
from clinicadl.config import config
from clinicadl.utils.config_utils import get_default_from_config_class as get_default
from clinicadl.utils.config_utils import get_type_from_config_class as get_type

emissions_calculator = click.option(
    "--calculate_emissions/--dont_calculate_emissions",
    default=get_default("emissions_calculator", config.CallbacksConfig),
    help="Flag to allow calculate the carbon emissions during training.",
    show_default=True,
)
track_exp = click.option(
    "--track_exp",
    "-te",
    type=get_type("track_exp", config.CallbacksConfig),
    default=get_default("track_exp", config.CallbacksConfig),
    help="Use `--track_exp` to enable wandb/mlflow to track the metric (loss, accuracy, etc...) during the training.",
    show_default=True,
)
