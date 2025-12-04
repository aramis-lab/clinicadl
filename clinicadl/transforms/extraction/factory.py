from typing import Any

from clinicadl.utils.factories import factory_from_dict

from .base import Extraction, ImplementedExtraction

# pylint: disable=unused-import
from .image import Image
from .patch import Patch
from .slice import Slice


@factory_from_dict(
    object_type=Extraction, enum=ImplementedExtraction, context=globals(), config=False
)
def get_extraction_from_dict(data: dict[str, Any]) -> Extraction:
    """
    Factory function to get a :py:class:`Extraction` from the
    dictionary returned by :py:meth:`Extraction.to_dict`.

    Parameters
    ----------
    data : dict[str, Any]
        The dictionary.

    Returns
    -------
    Extraction
        The config class, parametrized with the input dictionary.
    """
