import click

import clinicadl.train.trainer.training_config as config
from clinicadl.config import config
from clinicadl.utils.config_utils import get_default_from_config_class as get_default
from clinicadl.utils.config_utils import get_type_from_config_class as get_type

maps_dir = click.argument(
    "maps_dir", type=get_type("maps_dir", config.MapsManagerConfig)
)
data_group = click.option(
    "data_group", type=get_type("data_group", config.MapsManagerConfig)
)


overwrite = click.option(
    "--overwrite",
    "-o",
    is_flag=True,
    help="Will overwrite data group if existing. Please give caps_directory and participants_tsv to"
    " define new data group.",
)
save_nifti = click.option(
    "--save_nifti",
    is_flag=True,
    help="Save the output map(s) in the MAPS in NIfTI format.",
)
