from typing import Optional, Tuple, Union

import monai.metrics as metrics

from clinicadl.losses.utils import Loss
from clinicadl.utils.factories import DefaultFromLibrary, get_args_and_defaults

from .config.base import MetricConfig
from .config.enum import Reduction


def get_metric(config: MetricConfig) -> Tuple[metrics.Metric, MetricConfig]:
    """
    Factory function to get a metric from MONAI.

    Parameters
    ----------
    config : MetricConfig
        The config class with the parameters of the metric.

    Returns
    -------
    metrics.Metric
        The Metric object.
    MetricConfig
        The updated config class: the arguments set to default will be updated
        with their effective values (the default values from the library).
        Useful for reproducibility.
    """
    metric_class = getattr(metrics, config.metric)
    expected_args, config_dict = get_args_and_defaults(metric_class.__init__)
    for arg, value in config.model_dump().items():
        if arg in expected_args and value != DefaultFromLibrary.YES:
            config_dict[arg] = value

    metric = metric_class(**config_dict)
    updated_config = config.model_copy(update=config_dict)

    return metric, updated_config


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
        Defines mode of reduction. If not passed, the reduction method of the loss function will be used (if it exists).

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
