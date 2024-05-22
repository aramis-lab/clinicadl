import click

from clinicadl.config import config
from clinicadl.utils.config_utils import get_default_from_config_class as get_default
from clinicadl.utils.config_utils import get_type_from_config_class as get_type

# Computational
amp = click.option(
    "--amp/--no-amp",
    default=get_default("amp", config.ComputationalConfig),
    help="Enables automatic mixed precision during training and inference.",
    show_default=True,
)
fully_sharded_data_parallel = click.option(
    "--fully_sharded_data_parallel",
    "-fsdp",
    is_flag=True,
    help="Enables Fully Sharded Data Parallel with Pytorch to save memory at the cost of communications. "
    "Currently this only enables ZeRO Stage 1 but will be entirely replaced by FSDP in a later patch, "
    "this flag is already set to FSDP to that the zero flag is never actually removed.",
)
gpu = click.option(
    "--gpu/--no-gpu",
    default=get_default("gpu", config.ComputationalConfig),
    help="Use GPU by default. Please specify `--no-gpu` to force using CPU.",
    show_default=True,
)
