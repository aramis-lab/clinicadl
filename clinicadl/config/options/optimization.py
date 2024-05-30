import click

from clinicadl.config.config.optimization import OptimizationConfig
from clinicadl.utils.config_utils import get_default_from_config_class as get_default
from clinicadl.utils.config_utils import get_type_from_config_class as get_type

# Optimization
accumulation_steps = click.option(
    "--accumulation_steps",
    "-asteps",
    type=get_type("accumulation_steps", OptimizationConfig),
    default=get_default("accumulation_steps", OptimizationConfig),
    help="Accumulates gradients during the given number of iterations before performing the weight update "
    "in order to virtually increase the size of the batch.",
    show_default=True,
)
epochs = click.option(
    "--epochs",
    type=get_type("epochs", OptimizationConfig),
    default=get_default("epochs", OptimizationConfig),
    help="Maximum number of epochs.",
    show_default=True,
)
profiler = click.option(
    "--profiler/--no-profiler",
    default=get_default("profiler", OptimizationConfig),
    help="Use `--profiler` to enable Pytorch profiler for the first 30 steps after a short warmup. "
    "It will make an execution trace and some statistics about the CPU and GPU usage.",
    show_default=True,
)
