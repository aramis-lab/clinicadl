from copy import deepcopy
from typing import Any, Tuple, Union

import monai.metrics as metrics
from monai.metrics.metric import CumulativeIterationMetric as Metric

from clinicadl.utils.factories import update_config_with_defaults

from .config import ImplementedMetric, MetricConfig, create_metric_config


def get_metric_config(
    name: Union[str, ImplementedMetric],
    **kwargs: Any,
) -> MetricConfig:
    """
    Factory function to get a  metric configuration object from its name
    and parameters.

    Parameters
    ----------
    name : Union[str, ImplementedMetric]
        the name of the metric. Check our documentation to know available metrics.
    **kwargs : Any
        any parameter of the metric. Check our documentation on metrics to
        know these parameters.

    Returns
    -------
    MetricConfig
        the config object.
    """
    config = create_metric_config(name)(**kwargs)
    metric_class = getattr(metrics, config.name)

    update_config_with_defaults(config, function=metric_class.__init__)

    return config


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

    update_config_with_defaults(config, function=metric_class.__init__)

    config_dict = config.model_dump(exclude={"name"})
    metric = metric_class(**config_dict)

    return metric, config
