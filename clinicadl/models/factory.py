from enum import Enum

from clinicadl.utils.factories import factory_from_json
from clinicadl.utils.typing import PathType

# pylint: disable=unused-import
from .base import Model
from .reconstruction import ReconstructionModel
from .supervised import SupervisedModel


class ImplementedModel(str, Enum):
    """Implemented Models."""

    SUPERVISED = "SupervisedModel"
    RECONSTRUCTION = "ReconstructionModel"


@factory_from_json(
    object_type=Model, enum=ImplementedModel, context=globals(), config=False
)
def get_model_from_json(data: PathType) -> Model:
    """
    Factory function to get a :py:class:`Model` from the
    file saved with :py:meth:`Model.to_json`.

    Parameters
    ----------
    data : PathType
        The path to the ``json`` file.

    Returns
    -------
    Model
        The object, parametrized with the file content.
    """
