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
