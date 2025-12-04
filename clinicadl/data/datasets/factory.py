from enum import Enum
from typing import Any

from clinicadl.utils.factories import factory_from_dict, factory_from_json
from clinicadl.utils.typing import PathType

from .abstract import ClinicaDLDataset

# pylint: disable=unused-import
from .caps import CapsDataset
from .concat import ConcatDataset
from .paired import PairedDataset
from .unpaired import UnpairedDataset


class ImplementedDataset(str, Enum):
    """Implemented ClinicaDLDatasets."""

    CAPS = "CapsDataset"
    CONCAT = "ConcatDataset"
    PAIRED = "PairedDataset"
    UNPAIRED = "UnpairedDataset"


@factory_from_json(
    object_type=ClinicaDLDataset,
    enum=ImplementedDataset,
    context=globals(),
    config=False,
)
def get_dataset_from_json(data: PathType) -> ClinicaDLDataset:
    """
    Factory function to get a :py:class:`ClinicaDLDataset` from the
    file saved with :py:meth:`ClinicaDLDataset.to_json`.

    Parameters
    ----------
    data : PathType
        The path to the ``json`` file.

    Returns
    -------
    ClinicaDLDataset
        The object, parametrized with the file content.
    """


@factory_from_dict(
    object_type=ClinicaDLDataset,
    enum=ImplementedDataset,
    context=globals(),
    config=False,
)
def get_dataset_from_dict(data: dict[str, Any]) -> ClinicaDLDataset:
    """
    Factory function to get a :py:class:`ClinicaDLDataset` from the
    dictionary returned by :py:meth:`ClinicaDLDataset.to_dict`.

    Parameters
    ----------
    data : dict[str, Any]
        The dictionary.

    Returns
    -------
    ClinicaDLDataset
        The object, parametrized with the input dictionary.
    """
