from typing import Union

from .base import Metric
from .config import MetricConfig

MetricOrConfig = Union[Metric, MetricConfig]
