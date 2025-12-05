from __future__ import annotations

from typing import TYPE_CHECKING, Sequence, TypeVar

from clinicadl.utils.config import ObjectConfig
from clinicadl.utils.objects import HasConfig

from ..batch import Batch
from .base import CollateFn

if TYPE_CHECKING:
    from clinicadl.data.structures import DataPoint

T = TypeVar("T", bound="DataPoint")


class CollateSeqSamplesToBatchesConfig(ObjectConfig["CollateSeqSamplesToBatches"]):
    @classmethod
    def _get_class(cls) -> type[CollateSeqSamplesToBatches]:
        return CollateSeqSamplesToBatches


class CollateSeqSamplesToBatches(
    HasConfig[CollateSeqSamplesToBatchesConfig], CollateFn
):
    def __call__(self, datapoints: Sequence[Sequence[T]]) -> tuple[Batch[T], ...]:
        return tuple(Batch(data) for data in zip(*datapoints))
