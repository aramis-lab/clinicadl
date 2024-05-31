import click

from clinicadl.config.config.ssda import SSDAConfig
from clinicadl.config.config_utils import get_default_from_config_class as get_default
from clinicadl.config.config_utils import get_type_from_config_class as get_type

# SSDA
caps_target = click.option(
    "--caps_target",
    "-d",
    type=get_type("caps_target", SSDAConfig),
    default=get_default("caps_target", SSDAConfig),
    help="CAPS of target data.",
    show_default=True,
)
preprocessing_json_target = click.option(
    "--preprocessing_json_target",
    "-d",
    type=get_type("preprocessing_json_target", SSDAConfig),
    default=get_default("preprocessing_json_target", SSDAConfig),
    help="Path to json target.",
    show_default=True,
)
ssda_network = click.option(
    "--ssda_network/--single_network",
    default=get_default("ssda_network", SSDAConfig),
    help="If provided uses a ssda-network framework.",
    show_default=True,
)
tsv_target_lab = click.option(
    "--tsv_target_lab",
    "-d",
    type=get_type("tsv_target_lab", SSDAConfig),
    default=get_default("tsv_target_lab", SSDAConfig),
    help="TSV of labeled target data.",
    show_default=True,
)
tsv_target_unlab = click.option(
    "--tsv_target_unlab",
    "-d",
    type=get_type("tsv_target_unlab", SSDAConfig),
    default=get_default("tsv_target_unlab", SSDAConfig),
    help="TSV of unllabeled target data.",
    show_default=True,
)
