import click

from clinicadl.config.config.validation import ValidationConfig
from clinicadl.config.config_utils import get_default_from_config_class as get_default
from clinicadl.config.config_utils import get_type_from_config_class as get_type

# Validation
valid_longitudinal = click.option(
    "--valid_longitudinal/--valid_baseline",
    default=get_default("valid_longitudinal", ValidationConfig),
    help="If provided, not only the baseline sessions are used for validation (careful with this bad habit).",
    show_default=True,
)
evaluation_steps = click.option(
    "--evaluation_steps",
    "-esteps",
    type=get_type("evaluation_steps", ValidationConfig),
    default=get_default("evaluation_steps", ValidationConfig),
    help="Fix the number of iterations to perform before computing an evaluation. Default will only "
    "perform one evaluation at the end of each epoch.",
    show_default=True,
)

selection_metrics = click.option(
    "--selection_metrics",
    "-sm",
    type=get_type("selection_metrics", ValidationConfig),  # str list ?
    default=get_default("selection_metrics", ValidationConfig),  # ["loss"]
    multiple=True,
    help="""Allow to select a list of models based on their selection metric. Default will
    only infer the result of the best model selected on loss.""",
    show_default=True,
)
skip_leak_check = click.option(
    "--skip_leak_check",
    "-slc",
    is_flag=True,
    help="""Allow to skip the data leakage check usually performed. Not recommended.""",
)
