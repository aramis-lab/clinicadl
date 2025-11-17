from enum import Enum

from clinicadl.utils.factories import factory_from_json
from clinicadl.utils.typing import PathType

from .base import ClinicaDLModel
from .reconstruction import ReconstructionModel
from .supervised import SupervisedModel


class ImplementedModel(str, Enum):
    """Implemented ClinicaDLModels."""

    SUPERVISED = "SupervisedModel"
    RECONSTRUCTION = "ReconstructionModel"


@factory_from_json(
    object_type=ClinicaDLModel, enum=ImplementedModel, context=globals(), config=False
)
def get_model_from_json(data: PathType) -> ClinicaDLModel:
    """
    Factory function to get a :py:class:`ClinicaDLModel` from the
    file saved with :py:meth:`ClinicaDLModel.to_json`.

    Parameters
    ----------
    data : PathType
        The path to the ``json`` file.

    Returns
    -------
    ClinicaDLModel
        The object, parametrized with the file content.
    """
