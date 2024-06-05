import click

from clinicadl.config.config.reproducibility import ReproducibilityConfig
from clinicadl.config.config_utils import get_default_from_config_class as get_default
from clinicadl.config.config_utils import get_type_from_config_class as get_type

# Reproducibility
compensation = click.option(
    "--compensation",
    type=get_type("compensation", ReproducibilityConfig),
    default=get_default("compensation", ReproducibilityConfig),
    help="Allow the user to choose how CUDA will compensate the deterministic behaviour.",
    show_default=True,
)
deterministic = click.option(
    "--deterministic/--nondeterministic",
    default=get_default("deterministic", ReproducibilityConfig),
    help="Forces Pytorch to be deterministic even when using a GPU. "
    "Will raise a RuntimeError if a non-deterministic function is encountered.",
    show_default=True,
)
save_all_models = click.option(
    "--save_all_models/--save_only_best_model",
    type=get_type("save_all_models", ReproducibilityConfig),
    default=get_default("save_all_models", ReproducibilityConfig),
    help="If provided, enables the saving of models weights for each epochs.",
    show_default=True,
)
seed = click.option(
    "--seed",
    type=get_type("seed", ReproducibilityConfig),
    default=get_default("seed", ReproducibilityConfig),
    help="Value to set the seed for all random operations."
    "Default will sample a random value for the seed.",
    show_default=True,
)
config_file = click.option(
    "--config_file",
    "-c",
    type=get_type("config_file", ReproducibilityConfig),
    default=get_default("config_file", ReproducibilityConfig),
    help="Path to the TOML or JSON file containing the values of the options needed for training.",
)
