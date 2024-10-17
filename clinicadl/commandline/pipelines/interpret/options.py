import click

from clinicadl.config.config_utils import get_default_from_config_class as get_default
from clinicadl.config.config_utils import get_type_from_config_class as get_type
from clinicadl.interpret.config import InterpretBaseConfig

# interpret specific
name = click.argument(
    "name",
    type=get_type("name", InterpretBaseConfig),
)
method = click.argument(
    "method",
    type=get_type("method", InterpretBaseConfig),  # ["gradients", "grad-cam"]
)
level = click.option(
    "--level_grad_cam",
    type=get_type("level", InterpretBaseConfig),
    default=get_default("level", InterpretBaseConfig),
    help="level of the feature map (after the layer corresponding to the number) chosen for grad-cam.",
    show_default=True,
)
target_node = click.option(
    "--target_node",
    type=get_type("target_node", InterpretBaseConfig),
    default=get_default("target_node", InterpretBaseConfig),
    help="Which target node the gradients explain. Default takes the first output node.",
    show_default=True,
)
save_individual = click.option(
    "--save_individual",
    is_flag=True,
    help="Save individual saliency maps in addition to the mean saliency map.",
)
overwrite_name = click.option(
    "--overwrite_name",
    "-on",
    is_flag=True,
    help="Overwrite the name if it already exists.",
)
