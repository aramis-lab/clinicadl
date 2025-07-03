from typing import Union

from monai.metrics.metric import CumulativeIterationMetric as MonaiMetric

from clinicadl.losses.config import LossConfig
from clinicadl.losses.types import Loss
from clinicadl.metrics.config import CustomMetric, LossMetricConfig, MetricConfig

MetricType = Union[
    MetricConfig, MonaiMetric, CustomMetric, LossMetricConfig, LossConfig, Loss
]
