import click

from clinicadl.config.config_utils import get_default_from_config_class as get_default
from clinicadl.predictor.config import PredictConfig

# predict specific
use_labels = click.option(
    "--use_labels/--no_labels",
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
