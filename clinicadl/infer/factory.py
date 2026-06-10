from enum import Enum
from typing import Any

from clinicadl.utils.factories import factory_from_dict

# pylint: disable=unused-import
from .base import Inferer
from .patches_to_image import PatchesToImageInferer
from .simple import SimpleInferer
from .slices_to_image import SlicesToImageInferer


class ImplementedInferer(str, Enum):
    """Implemented Inferers."""

    SIMPLE = "SimpleInferer"
    PATCHES_TO_IMAGE = "PatchesToImageInferer"
    SLICES_TO_IMAGE = "SlicesToImageInferer"


@factory_from_dict(
    object_type=Inferer, enum=ImplementedInferer, context=globals(), config=False
)
def get_inferer_from_dict(data: dict[str, Any]) -> Inferer:
    """
    Factory function to get a :py:class:`Inferer` from the
    file saved with :py:meth:`Inferer.to_dict`.

    Parameters
    ----------
    data : dict[str, Any]
        The dictionary.

    Returns
    -------
    Inferer
        The inferer, parametrized with the input dictionary.
    """
