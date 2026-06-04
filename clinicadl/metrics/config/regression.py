import monai

from clinicadl.losses.config.enum import Reduction
from clinicadl.utils.doc import add_suffix_to_doc
from clinicadl.utils.factories import get_defaults_from

from ..enum import Optimum
from .base import DOCUMENT_EXTRA_PARAMETERS, MetricConfig, _GetNotNansConfig

__all__ = [
    "MSEMetricConfig",
    "MAEMetricConfig",
    "RMSEMetricConfig",
]

MSE_MONAI_DEFAULTS = get_defaults_from(monai.metrics.regression.MSEMetric)
MAE_MONAI_DEFAULTS = get_defaults_from(monai.metrics.regression.MAEMetric)
RMSE_MONAI_DEFAULTS = get_defaults_from(monai.metrics.regression.RMSEMetric)


@add_suffix_to_doc(DOCUMENT_EXTRA_PARAMETERS)
class MSEMetricConfig(MetricConfig, _GetNotNansConfig):
    """
    Config class for :py:class:`monai.metrics.MSEMetric`.

    ``get_not_nans`` is not supported currently.
    """

    reduction: Reduction = MSE_MONAI_DEFAULTS["reduction"]

    @staticmethod
    def optimum() -> Optimum:
        """The optimum of the metric."""
        return Optimum.MIN


@add_suffix_to_doc(DOCUMENT_EXTRA_PARAMETERS)
class MAEMetricConfig(MetricConfig, _GetNotNansConfig):
    """
    Config class for :py:class:`monai.metrics.MAEMetric`.

    ``get_not_nans`` is not supported currently.
    """

    reduction: Reduction = MAE_MONAI_DEFAULTS["reduction"]

    @staticmethod
    def optimum() -> Optimum:
        """The optimum of the metric."""
        return Optimum.MIN


@add_suffix_to_doc(DOCUMENT_EXTRA_PARAMETERS)
class RMSEMetricConfig(MetricConfig, _GetNotNansConfig):
    """
    Config class for :py:class:`monai.metrics.RMSEMetric`.

    ``get_not_nans`` is not supported currently.
    """

    reduction: Reduction = RMSE_MONAI_DEFAULTS["reduction"]

    @staticmethod
    def optimum() -> Optimum:
        """The optimum of the metric."""
        return Optimum.MIN
