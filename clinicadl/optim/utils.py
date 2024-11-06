from .lr_scheduler import LRSchedulerConfig
from .optimizer import OptimizerConfig


def check_optimizer_scheduler_consistency(
    optimizer_config: OptimizerConfig,
    lr_scheduler_config: LRSchedulerConfig,
) -> None:
    """
    Checks consistency between the optimizer and the LR scheduler configs.

    Parameters
    ----------
    optimizer_config : OptimizerConfig
        the configuration class for the optimizer.
    lr_scheduler_config : LRSchedulerConfig
        the configuration class for the LR scheduler.

    Raises
    ------
    ValueError
        If the parameter groups mentioned for the optimizer and the lr scheduler
        don't match.
    """
    optimizer_groups = optimizer_config.get_all_groups()
    scheduler_groups = lr_scheduler_config.get_all_groups()
    if len(scheduler_groups) > 0 and optimizer_groups != scheduler_groups:
        raise ValueError(
            "The parameter groups mentioned in optimizer config do not match "
            f"those mentioned in lr scheduler config. Got {optimizer_groups} in optimizer "
            f"config and {scheduler_groups} in lr scheduler config."
        )
