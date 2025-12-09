from __future__ import annotations

from typing import TYPE_CHECKING, Sequence, TypeVar

from clinicadl.utils.config import ObjectConfig
from clinicadl.utils.objects import HasConfig

from ..batch import Batch
from .base import CollateFn

if TYPE_CHECKING:
    from clinicadl.data.structures import Sample

T = TypeVar("T", bound="Sample")


class ToBatchesConfig(ObjectConfig["ToBatches"]):
    """
    Config class for ``ToBatches``.
    """

    @classmethod
    def _get_class(cls) -> type[ToBatches]:
        return ToBatches


class ToBatches(HasConfig[ToBatchesConfig], CollateFn):
    """
    To return a sequence of batches.

    This is the default collating mode when the :py:mod:`dataset <clinicadl.data.datasets` returns a sequence of samples.

    Examples
    --------

    .. code-block::

        from clinicadl.data.dataloader import ToBatches
        from clinicadl.data.structures.examples import ColinSample

        sample_1 = ColinSample(participant="sub-001")
        sample_2 = ColinSample(participant="sub-002")
        sample_3 = ColinSample(participant="sub-003")
        sample_4 = ColinSample(participant="sub-004")
        batch = ToBatches()([(sample_1, sample_2), (sample_3, sample_4)])

    .. code-block::

        >>> batch[0]
        [ColinSample(Keys: ('head', 'datatype', 'image_path', 'sample_type', 'sample_position', 'image', 'label', 'participant', 'session'); images: 3),
         ColinSample(Keys: ('head', 'datatype', 'image_path', 'sample_type', 'sample_position', 'image', 'label', 'participant', 'session'); images: 3)]
        >>> batch[0][0].participant
        'sub-001'
        >>> batch[0][1].participant
        'sub-003'

    See Also
    --------
    ~clinicadl.data.dataloader.MergeBatches
        To merge several batches into a single batch.
    """

    config = ToBatchesConfig()
    _config_type = ToBatchesConfig

    def __call__(self, samples: Sequence[Sequence[T]]) -> tuple[Batch[T], ...]:
        """
        Puts a batch of sequences of :py:class:`~clinicadl.data.datasets.Sample`
        in a tuple of :py:class:`~clinicadl.data.dataloader.Batch`.

        E.g. if a dataset returns two samples, the output here will be a tuple of two batches.

        Parameters
        ----------
        samples : Sequence[Sequence[T]]
            A sequence of sequences of :py:class:`~clinicadl.data.datasets.Sample`, e.g. a batch
            of outputs of a :py:class:`~clinicadl.data.datasets.PairedDataset`.

        Returns
        -------
        tuple[Batch[T], ...]
            A tuple of :py:class:`~clinicadl.data.dataloader.Batch`, whose dimension is equal
            to the 2nd dimension of ``samples``.
        """
        return tuple(Batch(data) for data in zip(*samples))
