# TODO: need to define what to put in the custom metrics

from abc import ABC, abstractmethod

from clinicadl.utils.config import ObjectConfig

from .enum import Optimum


class CustomMetric(ObjectConfig, ABC):
    """
    Abstract config class to use Custom metrics.
    The methods 'optimum' and '_get_class' must be overwritten.
    """

    @classmethod
    @abstractmethod
    def _get_class(cls):
        """Returns the metric associated to this config class."""
        pass

    @staticmethod
    @abstractmethod
    def optimum() -> Optimum:
        """The optimum of the metric."""
        raise NotImplementedError(
            "Optimum for Custom metrics is not defined. Please use the specific metric config class."
        )

    @property
    def name(self) -> str:
        """
        The name of the metric.
        If this method is not overwritten, the name will be CustomMetric.
        """
        return super().name
