import click

import clinicadl.train.trainer.training_config as config
from clinicadl.config import config
from clinicadl.utils.config_utils import get_default_from_config_class as get_default
from clinicadl.utils.config_utils import get_type_from_config_class as get_type

# predict specific
use_labels = click.option(
    "--use_labels/--no_labels",
    show_default=True,
    default=get_default("use_labels", config.PredictConfig),
    help="Set this option to --no_labels if your dataset does not contain ground truth labels.",
)
label = click.option(
    "--label",
    type=get_type("label", config.PredictConfig),
    default=get_default("label", config.PredictConfig),
    show_default=True,
    help="Target label used for training (if NETWORK_TASK in [`regression`, `classification`]). "
    "Default will reuse the same label as during the training task.",
)
save_tensor = click.option(
    "--save_tensor",
    is_flag=True,
    help="Save the reconstruction output in the MAPS in Pytorch tensor format.",
)
save_latent_tensor = click.option(
    "--save_latent_tensor",
    is_flag=True,
    help="""Save the latent representation of the image.""",
)
