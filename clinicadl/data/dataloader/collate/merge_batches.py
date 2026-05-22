from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Sequence, TypeVar

import numpy as np
import torch
import torchio as tio

from clinicadl.data.structures import Sample2D
from clinicadl.utils.config import ObjectConfig
from clinicadl.utils.dictionary.words import (
    FILE_TYPE,
    IMAGE_PATH,
    PARTICIPANT,
    SESSION,
    SQUEEZE,
)
from clinicadl.utils.numerics import merge_numerics
from clinicadl.utils.objects import HasConfig

from ..batch import Batch
from .base import CollateFn

if TYPE_CHECKING:
    from clinicadl.data.structures import Sample

T = TypeVar("T", bound="Sample")
FieldT = TypeVar("FieldT")


class MergeBatchesCollateConfig(ObjectConfig["MergeBatchesCollate"]):
    """
    Config class for ``MergeBatchesCollate``.
    """

    ignore: Optional[Sequence[str]]

    @classmethod
    def _get_class(cls) -> type[MergeBatchesCollate]:
        return MergeBatchesCollate


class MergeBatchesCollate(HasConfig[MergeBatchesCollateConfig], CollateFn):
    """
    To merge several batches into a single batch.

    This collating mode is typically to get a single batch from the outputs
    of a :py:mod:`dataset <clinicadl.data.datasets>` returning a sequence of samples.
    ``MergeBatchesCollate`` will try to merge this sequence of samples by merging each field
    of the samples, except those in ``ignore``.

    More precisely:
        - :py:class:`torchio.Images <torchio.Image>` will be concatenated along the channel dimension;
        - :py:class:`numpy.ndarrays <numpy.ndarray>` and :py:class:`torch.Tensors <torch.Tensors>`
            will be stacked along a new dimension;
        - otherwise, the values will be merged in a tuple. If this tuple contains only one unique
          element (i.e. the value is the same in all the input samples), a single value will be returned.

    Parameters
    ----------
    ignore : Optional[Sequence[str]], default=None
        To ignore some fields in the samples to merge. Thus, the output sample will not
        have these fields.

        .. important::
            The mandatory arguments of :py:class:`~clinicadl.data.datasets.Sample` cannot be
            ignored.

    Examples
    --------

    .. code-block::

        from clinicadl.data.dataloader import MergeBatchesCollate
        from clinicadl.data.structures.examples import ColinSample
        import numpy as np

        sample = ColinSample(participant="sub-001", label=np.array([0, 1]), age=55, sex="M")
        sample_bis = ColinSample(participant="sub-001", label=np.array([1, 2]), age=56, to_ignore="abc")

        batch = MergeBatchesCollate(ignore=["to_ignore"])([(sample, sample_bis)])

    .. code-block::

        >>> batch
        [ColinSample(Keys: ('head', 'sex', 'age', 'label', 'file_type', 'image_path', 'sample_type', 'sample_position', 'image', 'participant', 'session'); images: 2)]
        >>> batch[0].participant  # same value in the two samples
        'sub-001'
        >>> batch[0].age
        (55, 56)
        >>> batch[0].sex  # only in one sample, so kept as it is
        'M'
        >>> batch[0].image.shape  # 2 channels now!
        (2, 181, 217, 181)
        >>> batch[0].label  # arrays are stacked
        array([[0, 1],
               [1, 2]])

    See Also
    --------
    ~clinicadl.data.dataloader.ToBatchesCollate
        To return a sequence of batches.
    """

    _config_type = MergeBatchesCollateConfig

    def __init__(self, ignore: Optional[Sequence[str]] = None):
        self.config = MergeBatchesCollateConfig(ignore=ignore)

    def __call__(self, samples: Sequence[Sequence[T]]) -> Batch[T]:
        """
        Merges a batch of sequences of :py:class:`~clinicadl.data.datasets.Sample`
        in a single :py:class:`~clinicadl.data.dataloader.Batch`.

        Parameters
        ----------
        samples : Sequence[Sequence[T]]
            A sequence of sequences of :py:class:`~clinicadl.data.datasets.Sample`, e.g. a sequence
            of outputs of a :py:class:`~clinicadl.data.datasets.PairedDataset`.

        Returns
        -------
        Batch[T]
            A :py:class:`~clinicadl.data.dataloader.Batch`, whose :py:class:`Samples <clinicadl.data.datasets.Sample>`
            are the results of the merger of the inner input sequences.
        """
        mergers = []
        for samples_collection in samples:
            args = {}
            type_ = self._check_types(samples_collection)

            args[PARTICIPANT] = self._get_unique_field(
                [sample.participant for sample in samples_collection], PARTICIPANT
            )
            args[SESSION] = self._get_unique_field(
                [sample.session for sample in samples_collection], SESSION
            )
            if isinstance(samples_collection[0], Sample2D):
                args[SQUEEZE] = self._get_unique_field(
                    [sample[SQUEEZE] for sample in samples_collection],
                    SQUEEZE,
                )
            args[FILE_TYPE] = tuple(
                d for sample in samples_collection for d in sample.file_type
            )
            args[IMAGE_PATH] = tuple(
                p for sample in samples_collection for p in sample.image_path
            )

            for field in self._get_all_fields(samples_collection):
                if field in args or (
                    self.config.ignore and field in self.config.ignore
                ):
                    continue
                args[field] = self._merge_field(
                    [sample[field] for sample in samples_collection if field in sample],
                )

            mergers.append(type_(**args))

        return Batch(mergers)

    @staticmethod
    def _check_types(samples_collection: Sequence[Sample]) -> type:
        """
        Checks that the values of a field are consistent.
        """
        types = set(type(sample) for sample in samples_collection)
        if len(types) > 1:
            raise TypeError(f"Cannot merge samples of different types. Got {types}")

        return types.pop()

    @staticmethod
    def _get_unique_field(values: Sequence[FieldT], field_name: str) -> FieldT:
        """
        Checks that the values of a field are consistent.
        """
        unique_values = set(values)
        if len(unique_values) > 1:
            raise RuntimeError(
                f"Got different values for '{field_name}': {unique_values}"
            )

        return unique_values.pop()

    @staticmethod
    def _get_all_fields(samples_collection: Sequence[Sample]) -> set[str]:
        """
        Gets the list of all fields in a set of ``Samples``.
        """
        return set(field for sample in samples_collection for field in sample.keys())

    def _merge_field(self, values: Sequence[Any]) -> Any:
        """
        Tries to merge any field.
        """
        values = merge_numerics(values, merge_lists=False)

        if isinstance(values, (torch.Tensor, np.ndarray, tio.Image)):
            return values

        first = values[0]
        if all(value == first for value in values):
            return first
        else:
            return tuple(values)
