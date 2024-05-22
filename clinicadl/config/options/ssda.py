import click

import clinicadl.train.trainer.training_config as config
from clinicadl.config import config
from clinicadl.utils.config_utils import get_default_from_config_class as get_default
from clinicadl.utils.config_utils import get_type_from_config_class as get_type

# SSDA
caps_target = click.option(
    "--caps_target",
    "-d",
    type=get_type("caps_target", config.SSDAConfig),
    default=get_default("caps_target", config.SSDAConfig),
    help="CAPS of target data.",
    show_default=True,
)
preprocessing_json_target = click.option(
    "--preprocessing_json_target",
    "-d",
    type=get_type("preprocessing_json_target", config.SSDAConfig),
    default=get_default("preprocessing_json_target", config.SSDAConfig),
    help="Path to json target.",
    show_default=True,
)
ssda_network = click.option(
    "--ssda_network/--single_network",
    default=get_default("ssda_network", config.SSDAConfig),
    help="If provided uses a ssda-network framework.",
    show_default=True,
)
tsv_target_lab = click.option(
    "--tsv_target_lab",
    "-d",
    type=get_type("tsv_target_lab", config.SSDAConfig),
    default=get_default("tsv_target_lab", config.SSDAConfig),
    help="TSV of labeled target data.",
    show_default=True,
)
tsv_target_unlab = click.option(
    "--tsv_target_unlab",
    "-d",
    type=get_type("tsv_target_unlab", config.SSDAConfig),
    default=get_default("tsv_target_unlab", config.SSDAConfig),
    help="TSV of unllabeled target data.",
    show_default=True,
)
