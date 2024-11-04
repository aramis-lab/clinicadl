from copy import deepcopy
from typing import Any, Optional, Tuple, Union

import monai.metrics as metrics
from monai.metrics.metric import CumulativeIterationMetric as Metric

from clinicadl.losses.utils import Loss
from clinicadl.utils.factories import DefaultFromLibrary, get_args_and_defaults

from .config import ImplementedMetric, MetricConfig, create_metric_config
from .config.enum import Reduction


def get_metric(
    name: Union[str, ImplementedMetric], return_config: bool = False, **kwargs: Any
) -> Union[Metric, Tuple[Metric, MetricConfig]]:
    """
    Factory function to get a MONAI metric from its name and parameters.

    Parameters
    ----------
    name : Union[str, ImplementedMetric]
        the name of the metric. Check our documentation to know available metrics.
    return_config : bool (optional, default=False)
        if the function should return the config class regrouping the parameters of the
        metric. Useful to keep track of the hyperparameters.
    **kwargs : Any
        the parameters of the metric. Check our documentation on metrics to
        know these parameters.

    Returns
    -------
    Metric
        the metric.
    MetricConfig
        the associated config object. Only returned if `return_config` is True.
    """
    config = create_metric_config(name)(**kwargs)
    metric, updated_config = get_metric_from_config(config)

    return metric if not return_config else (metric, updated_config)


def get_metric_from_config(config: MetricConfig) -> Tuple[Metric, MetricConfig]:
    """
    Factory function to get a MONAI metric from a MetricConfig instance.

    Parameters
    ----------
    config : MetricConfig
        the configuration object.

    Returns
    -------
    Metric
        the metric.
    MetricConfig
        the updated config class: the arguments set to default will be updated
        with their effective values (the default values from the metric).
        Useful for reproducibility.
    """
    config = deepcopy(config)
    metric_class = getattr(metrics, config.name)

    # update config with defaults
    _, defaults = get_args_and_defaults(metric_class.__init__)
    for arg, value in config:
        if value == DefaultFromLibrary.YES and arg in defaults:
            setattr(config, arg, defaults[arg])

    config_dict = config.model_dump(exclude={"name"})
    metric = metric_class(**config_dict)

    return metric, config


def loss_to_metric(
    loss_fn: Loss,
    reduction: Optional[Union[str, Reduction]] = None,
) -> metrics.LossMetric:
    """
    Converts a loss function to a metric object.

    Parameters
    ----------
    loss_fn : Loss
        A callable function that takes y_pred and optionally y as input (in the “batch-first” format), returns a 1-item tensor.
        loss_fn can also be a PyTorch loss object.
    reduction : Optional[Union[str, Reduction]] (optional, default=None)
        Defines mode of reduction. If not passed, the reduction method of the loss function will be used.

    Returns
    -------
    metrics.LossMetric
        The loss function wrapped in a metric object.

    Raises
    ------
    ValueError
        If the user didn't pass a reduction method, and the loss function doesn't have an attribute 'reduction'.
    """
    if reduction is None:
        try:
            checked_reduction = loss_fn.reduction
        except AttributeError as exc:
            raise ValueError(
                "If the loss function doesn't have an attribute 'reduction', you must pass a reduction method."
            ) from exc
    else:
        checked_reduction = Reduction(reduction)

    return metrics.LossMetric(loss_fn, reduction=checked_reduction)
