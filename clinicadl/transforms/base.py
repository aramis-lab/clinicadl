from abc import ABC, abstractmethod

from clinicadl.data.structures import DataPoint


class BaseTransform(ABC):
    """
    Transforms should follow this pattern to work with ``ClinicaDL``.
    """

    @abstractmethod
    def __call__(self, datapoint: DataPoint) -> DataPoint:
        """
        The transform must take as argument and return a :py:class:`clinicadl.data.structures.DataPoint`.
        """
