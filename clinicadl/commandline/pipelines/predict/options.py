import click

from clinicadl.config.config.pipelines.predict import PredictConfig
from clinicadl.config.config_utils import get_default_from_config_class as get_default
from clinicadl.config.config_utils import get_type_from_config_class as get_type

# predict specific
use_labels = click.option(
    "--use_labels/--no_labels",
    show_default=True,
    default=get_default("use_labels", PredictConfig),
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
