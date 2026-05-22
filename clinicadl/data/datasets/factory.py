from enum import Enum
from typing import Any, Optional

from clinicadl.io.bids.reader import Bids
from clinicadl.utils.factories import (
    factory_from_dict,
    factory_from_json,
    safe_factory_from_json,
)
from clinicadl.utils.typing import PathType

from .base import Dataset

# pylint: disable=unused-import
from .bids import BidsDataset
from .concat import ConcatDataset
from .paired import PairedDataset
from .tensor import TensorDataset
from .unpaired import UnpairedDataset

ImplementedDatasetT = (
    BidsDataset | TensorDataset | ConcatDataset | PairedDataset | UnpairedDataset
)


class ImplementedDataset(str, Enum):
    """Implemented Datasets."""

    BIDS = "BidsDataset"
    TENSOR = "TensorDataset"
    CONCAT = "ConcatDataset"
    PAIRED = "PairedDataset"
    UNPAIRED = "UnpairedDataset"


@factory_from_json(
    object_type=Dataset,
    enum=ImplementedDataset,
    context=globals(),
    config=False,
)
def get_dataset_from_json(data: PathType) -> Dataset:
    """
    Factory function to get a :py:class:`Dataset` from the
    file saved with :py:meth:`Dataset.to_json`.

    Parameters
    ----------
    data : PathType
        The path to the ``json`` file.

    Returns
    -------
    Dataset
        The object, parametrized with the file content.
    """


@factory_from_dict(
    object_type=Dataset,
    enum=ImplementedDataset,
    context=globals(),
    config=False,
)
def get_dataset_from_dict(data: dict[str, Any]) -> Dataset:
    """
    Factory function to get a :py:class:`Dataset` from the
    dictionary returned by :py:meth:`Dataset.to_dict`.

    Parameters
    ----------
    data : dict[str, Any]
        The dictionary.

    Returns
    -------
    Dataset
        The object, parametrized with the input dictionary.
    """


@safe_factory_from_json(factory=get_dataset_from_json)
def get_dataset_from_json_safely(
    json_path: PathType, default: Optional[Dataset] = None
) -> tuple[Optional[Dataset], list[str]]:
    """
    Factory function to get a :py:class:`Dataset` from the
    file saved with :py:meth:`Dataset.to_json` that will not raised errors.

    If some fields of the serialized dataset cannot be read, they will be reported, and
    the field of ``default`` will be used to override them (if not ``None``).

    If it was impossible to read the serialized dataset, the factory returns ``None``.

    Parameters
    ----------
    json_path : PathType
        The path to the serialized dataset.
    default : Optional[Dataset], default=None
        The :py:class:`Dataset` from which to take the default arguments.

    Returns
    -------
    Optional[Dataset]
        The deserialized dataset. ``None`` if deserialization was impossible.
    list[str]
        The list of fields that could not be read in the serialized dataset.
    """
