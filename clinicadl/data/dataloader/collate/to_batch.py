from __future__ import annotations

from typing import TYPE_CHECKING, Sequence, TypeVar

from clinicadl.utils.config import ObjectConfig
from clinicadl.utils.objects import HasConfig

from ..batch import Batch
from .base import CollateFn

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


class ToBatchCollate(HasConfig[ToBatchCollateConfig], CollateFn):
    """
    To simply collate a sequence of samples in a single batch.

    This is the default collating mode when the :py:mod:`dataset <clinicadl.data.datasets` returns a single sample.

    Examples
    --------

    .. code-block::

        from clinicadl.data.dataloader import ToBatchCollate
        from clinicadl.data.structures.examples import ColinSample

        sample_1 = ColinSample(participant="sub-001")
        sample_2 = ColinSample(participant="sub-002")
        batch = ToBatchCollate()([sample_1, sample_2])

    .. code-block::

        >>> batch
        [ColinSample(Keys: ('head', 'file_type', 'image_path', 'sample_type', 'sample_position', 'image', 'participant', 'session'); images: 3),
         ColinSample(Keys: ('head', 'file_type', 'image_path', 'sample_type', 'sample_position', 'image', 'participant', 'session'); images: 3)]
        >>> batch[0].participant
        'sub-001'
        >>> batch[1].participant
        'sub-002'
    """

    config = ToBatchCollateConfig()
    _config_type = ToBatchCollateConfig

    def __call__(self, samples: Sequence[T]) -> Batch[T]:
        """
        Puts a sequence of :py:class:`~clinicadl.data.datasets.Sample` in a :py:class:`~clinicadl.data.dataloader.Batch`.

        Parameters
        ----------
        samples : Sequence[T]
            A sequence :py:class:`~clinicadl.data.datasets.Sample`.

        Returns
        -------
        Batch[T]
            A :py:class:`~clinicadl.data.dataloader.Batch`.
        """
        return Batch(samples)
