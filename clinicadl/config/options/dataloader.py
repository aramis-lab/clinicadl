import click

import clinicadl.train.trainer.training_config as config
from clinicadl.config import config
from clinicadl.utils.config_utils import get_default_from_config_class as get_default
from clinicadl.utils.config_utils import get_type_from_config_class as get_type

# DataLoader
batch_size = click.option(
    "--batch_size",
    type=get_type("batch_size", config.DataLoaderConfig),
    default=get_default("batch_size", config.DataLoaderConfig),
    help="Batch size for data loading.",
    show_default=True,
)
n_proc = click.option(
    "-np",
    "--n_proc",
    type=get_type("n_proc", config.DataLoaderConfig),
    default=get_default("n_proc", config.DataLoaderConfig),
    help="Number of cores used during the task.",
    show_default=True,
)
sampler = click.option(
    "--sampler",
    "-s",
    type=get_type("sampler", config.DataLoaderConfig),
    default=get_default("sampler", config.DataLoaderConfig),
    help="Sampler used to load the training data set.",
    show_default=True,
)
