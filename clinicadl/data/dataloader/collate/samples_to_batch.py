from __future__ import annotations

from typing import TYPE_CHECKING, Sequence, TypeVar

from clinicadl.utils.config import ObjectConfig
from clinicadl.utils.objects import HasConfig

from ..batch import Batch
from .base import CollateFn

if TYPE_CHECKING:
    from clinicadl.data.structures import DataPoint

T = TypeVar("T", bound="DataPoint")


class CollateSamplesToBatchCollateConfig(ObjectConfig["CollateSamplesToBatchCollate"]):
    @classmethod
    def _get_class(cls) -> type[CollateSamplesToBatchCollate]:
        return CollateSamplesToBatchCollate


class CollateSamplesToBatchCollate(
    HasConfig[CollateSamplesToBatchCollateConfig], CollateFn
):
    def __call__(self, samples: Sequence[T]) -> Batch[T]:
        return Batch(samples)
