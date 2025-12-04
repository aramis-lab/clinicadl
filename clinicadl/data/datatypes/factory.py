from enum import Enum
from typing import Any

from clinicadl.utils.factories import factory_from_dict

from .base import DataType
from .preprocessing import *

SupportedDataType = Enum(
    "SupportedDataType", {**SupportedPreprocessing.__members__, "BASE": "DataType"}
)


@factory_from_dict(
    object_type=DataType, enum=SupportedDataType, context=globals(), config=False
)
def get_datatype_from_dict(data: dict[str, Any]) -> DataType:
    """
    Factory function to get a :py:class:`DataType` from the
    dictionary returned by :py:meth:`DataType.to_dict`.

    Parameters
    ----------
    data : dict[str, Any]
        The dictionary.

    Returns
    -------
    DataType
        The config class, parametrized with the input dictionary.
    """
