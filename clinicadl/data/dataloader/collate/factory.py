from enum import Enum
from typing import Any

from clinicadl.utils.factories import factory_from_dict

from .base import CollateFn
from .merge_batches import MergeBatchesCollate
from .to_batch import ToBatchCollate
from .to_batches import ToBatchesCollate


class ImplementedCollateFn(str, Enum):
    """
    Collate mode supported natively in ``ClinicaDL``.
    """

    TO_BATCH = "ToBatchCollate"
    TO_BATCHES = "ToBatchesCollate"
    MERGE_BATCH = "MergeBatchesCollate"


@factory_from_dict(
    object_type=CollateFn, enum=ImplementedCollateFn, context=globals(), config=False
)
def get_collate_from_dict(data: dict[str, Any]) -> CollateFn:
    """
    Factory function to get a :py:class:`CollateFn` from the
    dictionary returned by :py:meth:`CollateFn.to_dict`.

    Parameters
    ----------
    data : dict[str, Any]
        The dictionary.

    Returns
    -------
    CollateFn
        The collate callable, parametrized with the input dictionary.
    """
