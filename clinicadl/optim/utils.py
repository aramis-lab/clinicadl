from .lr_schedulers.config import (
    LRSchedulerConfig,
    OneCycleLRConfig,
)
from .optimizers.config import (
    AdamConfig,
    OptimizerConfig,
    RMSpropConfig,
    SGDConfig,
)


def check_optimizer_scheduler_consistency(
    optimizer_config: OptimizerConfig,
    lr_scheduler_config: LRSchedulerConfig,
) -> None:
    """
    Checks consistency between the optimizer and the LR scheduler configs.

    Parameters
    ----------
    optimizer_config : OptimizerConfig
        The configuration class for the optimizer.
    lr_scheduler_config : LRSchedulerConfig
        The configuration class for the LR scheduler.

    Raises
    ------
    ValueError
        If the LR scheduler is 'OneCycleLR' with 'cycle_momentum=True' and the optimizer
        does not have a momentum.
    ValueError
        If the parameter groups mentioned for the optimizer and the lr scheduler
        don't match.
    """
    if (
        isinstance(lr_scheduler_config, OneCycleLRConfig)
        and lr_scheduler_config.cycle_momentum
        and not isinstance(OptimizerConfig, (SGDConfig, RMSpropConfig, AdamConfig))
    ):
        raise ValueError(
            "If 'cycle_momentum' is True in OneCycleLR, the optimizer can't be "
            f"{optimizer_config.name} because it requires a momentum."
        )

    optimizer_groups = optimizer_config.get_all_groups()
    scheduler_groups = lr_scheduler_config.get_all_groups()
    if len(scheduler_groups) > 0 and optimizer_groups != scheduler_groups:
        raise ValueError(
            "The parameter groups mentioned in optimizer config do not match "
            f"those mentioned in lr scheduler config. Got {optimizer_groups} in optimizer "
            f"config and {scheduler_groups} in lr scheduler config."
        )
