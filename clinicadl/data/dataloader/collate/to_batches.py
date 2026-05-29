from __future__ import annotations

from typing import TYPE_CHECKING, Sequence, TypeVar

from clinicadl.utils.config import ObjectConfig
from clinicadl.utils.objects import HasConfig

from ..batch import Batch
from .base import ImplementedCollateFn

if TYPE_CHECKING:
    from clinicadl.data.structures import Sample

T = TypeVar("T", bound="Sample")


class ToBatchesCollateConfig(ObjectConfig["ToBatchesCollate"]):
    """
    Config class for ``ToBatchesCollate``.
    """

    @classmethod
    def _get_class(cls) -> type[ToBatchesCollate]:
        return ToBatchesCollate


class ToBatchesCollate(ImplementedCollateFn, HasConfig[ToBatchesCollateConfig]):
    """
    To return a sequence of batches.

    This is the default collating mode when the :py:class:`~clinicadl.data.datasets.Dataset` returns a sequence of
    :py:class:`~clinicadl.data.structures.Sample`.

    Examples
    --------

    .. code-block::

        from clinicadl.data.dataloader import ToBatchesCollate
        from clinicadl.data.structures.examples import Colin27Sample

        sample_1 = Colin27Sample(participant="sub-001")
        sample_2 = Colin27Sample(participant="sub-002")
        sample_3 = Colin27Sample(participant="sub-003")
        sample_4 = Colin27Sample(participant="sub-004")
        batch = ToBatchesCollate()([(sample_1, sample_2), (sample_3, sample_4)])

    .. code-block::

        >>> batch[0]
        [Colin27Sample(Keys: ('head', 'file_type', 'image_path', 'sample_type', 'sample_position', 'image', 'participant', 'session'); images: 2),
         Colin27Sample(Keys: ('head', 'file_type', 'image_path', 'sample_type', 'sample_position', 'image', 'participant', 'session'); images: 2)]
        >>> batch[0][0].participant
        'sub-001'
        >>> batch[0][1].participant
        'sub-003'

    See Also
    --------
    ~clinicadl.data.dataloader.MergeBatchesCollate
        To merge several batches into a single batch.
    """

    config = ToBatchesCollateConfig()
    _config_type = ToBatchesCollateConfig

    def __call__(self, samples: Sequence[Sequence[T]]) -> tuple[Batch[T], ...]:
        """
        Puts a sequence of sequences of :py:class:`~clinicadl.data.structures.Sample`
        in a tuple of :py:class:`~clinicadl.data.dataloader.Batch`.

        E.g., if the dataset returns two samples, the output here will be a tuple of two batches.

        Parameters
        ----------
        samples : Sequence[Sequence[T]]
            A sequence of sequences of :py:class:`~clinicadl.data.structures.Sample`.

        Returns
        -------
        tuple[Batch[T], ...]
            A tuple of :py:class:`~clinicadl.data.dataloader.Batch`, whose dimension is equal
            to the 2nd dimension of ``samples``.
        """
        return tuple(Batch(data) for data in zip(*samples))
