from __future__ import annotations

from collections import Counter
from copy import deepcopy
from typing import TYPE_CHECKING, Optional, Sequence, TypeVar

import torchio as tio

from clinicadl.utils.config import ObjectConfig
from clinicadl.utils.objects import HasConfig

from ..batch import Batch
from .base import CollateFn

if TYPE_CHECKING:
    from clinicadl.data.structures import DataPoint


T = TypeVar("T", bound="DataPoint")


class CollateSeqSamplesToBatchCollateConfig(
    ObjectConfig["CollateSeqSamplesToBatchCollate"]
):
    @classmethod
    def _get_class(cls) -> type[CollateSeqSamplesToBatchCollate]:
        return CollateSeqSamplesToBatchCollate


class CollateSeqSamplesToBatchCollate(
    HasConfig[CollateSeqSamplesToBatchCollateConfig], CollateFn
):
    def __init__(self, merge_only: Optional[Sequence[str]]):
        self.merge_only = merge_only

    def __call__(self, samples: Sequence[Sequence[T]]) -> Batch[T]:
        for samples_collection in samples:
            types = set(type(sample) for sample in samples_collection)
            if len(types) > 1:
                raise TypeError("")

            merger_result = deepcopy(samples[0])
            all_fields = set(sample.keys() for sample in samples_collection)
            for field in all_fields:
                if self.merge_only and field not in self.merge_only:
                    continue

    @classmethod
    def _check_fields(cls, samples_collection: Sequence[T]) -> None:
        common_fields = cls._get_common_fields(samples_collection)
        for sample in samples_collection:
            for field in common_fields:
                try:
                    value = sample[field]
                except KeyError:
                    continue
                if not isinstance(value, tio.Image):
                    raise ValueError(
                        f"Cannot merge fields that are not torchio.Image. Got {value} for {field} in one of the samples."
                    )

    @staticmethod
    def _get_common_fields(samples_collection: Sequence[T]) -> set[str]:
        all_elements = [
            field for sample in samples_collection for field in sample.keys()
        ]
        counts = Counter(all_elements)
        return {elem for elem, count in counts.items() if count >= 2}
