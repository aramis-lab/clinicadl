from typing import Optional, Union

import torchio as tio
from pydantic import (
    computed_field,
)

from clinicadl.utils.config import DefaultFromLibrary

from ...config import TransformConfig
from ..nan_removal import NanRemoval
from .enum import ZooTransform


class NanRemovalConfig(TransformConfig):
    """
    Config class for :py:class:`clinicadl.transforms.NanRemoval <clinicadl.transforms.homemade_transforms.NanRemoval>`.
    """

    nan: float
    posinf: Optional[float]
    neginf: Optional[float]

    def __init__(
        self,
        nan: Union[float, DefaultFromLibrary] = DefaultFromLibrary.YES,
        posinf: Union[Optional[float], DefaultFromLibrary] = DefaultFromLibrary.YES,
        neginf: Union[Optional[float], DefaultFromLibrary] = DefaultFromLibrary.YES,
    ):
        super().__init__(nan=nan, posinf=posinf, neginf=neginf)

    @computed_field
    @property
    def name(self) -> str:
        """The name of the transform."""
        return ZooTransform.NAN_REMOVAL.value

    def _get_class(self) -> type[tio.Transform]:
        """Returns the transform associated to this config class."""
        return NanRemoval
