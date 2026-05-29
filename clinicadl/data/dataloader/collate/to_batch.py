from __future__ import annotations

from typing import TYPE_CHECKING, Sequence, TypeVar

from clinicadl.utils.config import ObjectConfig
from clinicadl.utils.objects import HasConfig

from ..batch import Batch
from .base import ImplementedCollateFn

if TYPE_CHECKING:
    from clinicadl.data.structures import Sample

T = TypeVar("T", bound="Sample")


class ToBatchCollateConfig(ObjectConfig["ToBatchCollate"]):
    """
    Config class for ``ToBatchCollate``.
    """

    @classmethod
    def _get_class(cls) -> type[ToBatchCollate]:
        return ToBatchCollate


class ToBatchCollate(ImplementedCollateFn, HasConfig[ToBatchCollateConfig]):
    """
    To simply collate a sequence of samples in a single batch.

    This is the default collating mode when the :py:class:`~clinicadl.data.datasets.Dataset` returns a single :py:class:`~clinicadl.data.structures.Sample`.

    Examples
    --------

    .. code-block::

        from clinicadl.data.dataloader import ToBatchCollate
        from clinicadl.data.structures.examples import Colin27Sample

        sample_1 = Colin27Sample(participant="sub-001")
        sample_2 = Colin27Sample(participant="sub-002")
        batch = ToBatchCollate()([sample_1, sample_2])

    .. code-block::

        >>> batch
        [Colin27Sample(Keys: ('head', 'file_type', 'image_path', 'sample_type', 'sample_position', 'image', 'participant', 'session'); images: 3),
         Colin27Sample(Keys: ('head', 'file_type', 'image_path', 'sample_type', 'sample_position', 'image', 'participant', 'session'); images: 3)]
        >>> batch[0].participant
        'sub-001'
        >>> batch[1].participant
        'sub-002'
    """

    config = ToBatchCollateConfig()
    _config_type = ToBatchCollateConfig

    def __call__(self, samples: Sequence[T]) -> Batch[T]:
        """
        Puts a sequence of :py:class:`~clinicadl.data.structures.Sample` in a :py:class:`~clinicadl.data.dataloader.Batch`.

        Parameters
        ----------
        samples : Sequence[T]
            A sequence of :py:class:`~clinicadl.data.structures.Sample`.

        Returns
        -------
        Batch[T]
            A :py:class:`~clinicadl.data.dataloader.Batch`.
        """
        return Batch(samples)
