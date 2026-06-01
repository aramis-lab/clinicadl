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
    Abstract class to define how sequences of :py:class:`~clinicadl.data.structures.Sample`
    are collated into batches.

    See :torch:`PyTorch's documentation <data.html#loading-batched-and-non-batched-data>`.

    The only function to override is :py:meth:`__call__`, which defines how
    the samples are collated, and thus what will be the output of the :py:class:`~clinicadl.data.dataloader.DataLoader`.
    """

    @abstractmethod
    def __call__(self, samples: Sequence[SampleLike]) -> BatchType:
        """
        Defines how the samples are collated.

        Parameters
        ----------
        samples : Sequence[SampleLike]
            A sequence of :py:class:`Samples <clinicadl.data.structures.Sample>`, sequences of ``Samples``, or
            dictionaries of ``Samples``.

        Returns
        -------
        BatchType
            A :py:class:`~clinicadl.data.dataloader.Batch`, a sequence of ``Batches``, or a dictionary of ``Batches``.
        """


@equal_if_config_equal
class ImplementedCollateFn(CollateFn, HasConfig):
    pass
