from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Sequence, TypeVar, Union

from clinicadl.utils.objects import HasConfig, JsonReaderWriter, equal_if_config_equal

from ..batch import Batch

if TYPE_CHECKING:
    from clinicadl.data.structures import Sample

T = TypeVar("T", bound="Sample")
SampleLike = Union[T, Sequence[T], dict[Any, T]]
BatchType = Union[Batch[T], Sequence[Batch[T]], dict[Any, Batch[T]]]


class CollateFn(JsonReaderWriter, ABC):
    """
    Abstract class to define how list of :py:class:`~clinicadl.data.datasets.Sample`
    are collated into batches.

    See :torch:`PyTorch's documentation <data.html#loading-batched-and-non-batched-data>`.

    The only function to override is :py:meth:`__call__`, which defines how
    the samples are collated, and thus what will be the output of the ``DataLoader``.
    """

    @abstractmethod
    def __call__(self, samples: Sequence[SampleLike]) -> BatchType:
        """
        Defines how the samples are collated.

        Parameters
        ----------
        samples : Sequence[SampleLike]
            A sequence :py:class:`~clinicadl.data.datasets.Sample`, a sequence of sequences ``Samples``, or
            a sequence of dictionaries of ``Samples``.

        Returns
        -------
        BatchType
            A :py:class:`~clinicadl.data.dataloader.Batch`, a sequence of ``Batches``, or a dictionary of ``Batches``.
        """


@equal_if_config_equal
class ImplementedCollateFn(CollateFn, HasConfig):
    pass
