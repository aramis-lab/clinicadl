from __future__ import annotations

from typing import TYPE_CHECKING, Sequence, TypeVar

from clinicadl.utils.config import ObjectConfig
from clinicadl.utils.objects import HasConfig

from ..batch import Batch
from .base import CollateFn

if TYPE_CHECKING:
    from clinicadl.data.structures import DataPoint

T = TypeVar("T", bound="DataPoint")


class CollateSamplesToBatchConfig(ObjectConfig["CollateSamplesToBatch"]):
    @classmethod
    def _get_class(cls) -> type[CollateSamplesToBatch]:
        return CollateSamplesToBatch


class CollateSamplesToBatch(HasConfig[CollateSamplesToBatchConfig], CollateFn):
    def __call__(self, samples: Sequence[T]) -> Batch[T]:
        return Batch(samples)
