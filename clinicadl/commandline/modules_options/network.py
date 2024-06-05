import click

from clinicadl.config.config_utils import get_default_from_config_class as get_default
from clinicadl.config.config_utils import get_type_from_config_class as get_type
from clinicadl.network.config import NetworkConfig

# Model
multi_network = click.option(
    "--multi_network/--single_network",
    default=get_default("multi_network", NetworkConfig),
    help="If provided uses a multi-network framework.",
    show_default=True,
)
dropout = click.option(
    "--dropout",
    type=get_type("dropout", NetworkConfig),
    default=get_default("dropout", NetworkConfig),
    help="Rate value applied to dropout layers in a CNN architecture.",
    show_default=True,
)
