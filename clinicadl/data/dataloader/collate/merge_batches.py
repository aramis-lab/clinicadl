from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Sequence, TypeVar

import numpy as np
import torch
import torchio as tio

from clinicadl.utils.config import ObjectConfig
from clinicadl.utils.dictionary.words import (
    DATATYPE,
    IMAGE,
    IMAGE_PATH,
    LABEL,
    PARTICIPANT,
    SESSION,
)
from clinicadl.utils.factories import get_args_from
from clinicadl.utils.objects import HasConfig
from clinicadl.utils.variables import SPACING_RTOL

from ..batch import Batch
from .base import CollateFn

if TYPE_CHECKING:
    from clinicadl.data.structures import Sample

T = TypeVar("T", bound="Sample")
FieldT = TypeVar("FieldT")


class MergeBatchesConfig(ObjectConfig["MergeBatches"]):
    """
    Config class for ``MergeBatches``.
    """

    ignore: Optional[Sequence[str]]

    @classmethod
    def _get_class(cls) -> type[MergeBatches]:
        return MergeBatches


class MergeBatches(HasConfig[MergeBatchesConfig], CollateFn):
    """
    To merge several batches into a single batch.

    This collating mode is typically to get a single batch from the outputs
    of a :py:mod:`dataset <clinicadl.data.datasets` returning a sequence of samples.
    ``MergeBatches`` will try to merge this sequence of samples by merging each field
    of the samples, except those in ``ignore``.

    More precisely, images will be concatenated along the channel dimension and numeric
    sequences will be stacked along a new dimension.

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

        from clinicadl.data.dataloader import MergeBatches
        from clinicadl.data.structures.examples import ColinSample
        import numpy as np

        sample_1 = ColinSample(participant="sub-001", extra=np.array([0, 1]), extra_bis=0)
        sample_1_bis = ColinSample(participant="sub-001", extra=np.array([1, 2]), to_ignore="abc")
        sample_2 = ColinSample()
        sample_2_bis = ColinSample()

        batch = MergeBatches(ignore=["to_ignore"])([(sample_1, sample_1_bis), (sample_2, sample_2_bis)])

    .. code-block::

        >>> batch
        [ColinSample(Keys: ('head', 'extra', 'extra_bis', 'datatype', 'image_path', 'sample_type', 'sample_position', 'image', 'label', 'participant', 'session'); images: 3),
         ColinSample(Keys: ('head', 'datatype', 'image_path', 'sample_type', 'sample_position', 'image', 'label', 'participant', 'session'); images: 3)]
        >>> batch[0].participant
        'sub-001'
        >>> batch[0].image.shape  # 2 channels now!
        (2, 181, 217, 181)
        >>> batch[0].extra  # arrays are stacked
        array([[0, 1],
               [1, 2]])
        >>> batch[0].extra_bis  # only in one sample, so kept as it is
        0

    See Also
    --------
    ~clinicadl.data.dataloader.ToBatches
        To return a sequence of batches.
    """

    _config_type = MergeBatchesConfig

    def __init__(self, ignore: Optional[Sequence[str]] = None):
        self.config = MergeBatchesConfig(ignore=ignore)

    def __call__(self, samples: Sequence[Sequence[T]]) -> Batch[T]:
        """
        Merges a batch of sequences of :py:class:`~clinicadl.data.datasets.Sample`
        in a single :py:class:`~clinicadl.data.dataloader.Batch`.

        More precisely:
            - :py:class:`torchio.Images <torchio.Image>` will be concatenated along the channel dimension;
            - :py:class:`numpy.ndarrays <numpy.ndarray>` and :py:class:`torch.Tensors <torch.Tensors>`
              will be stacked along a new dimension;
            - ``MergeBatches`` doesn't support the merger of other types of data. So, if a field of the :py:class:`Samples <~clinicadl.data.datasets.Sample>` contains
              other type of data, the merger will be successful only if a single value is passed (see examples).

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

            args[IMAGE] = self._merge_field(
                [sample.image for sample in samples_collection], IMAGE
            )
            args[PARTICIPANT] = self._get_unique_field(
                [sample.participant for sample in samples_collection], PARTICIPANT
            )
            args[SESSION] = self._get_unique_field(
                [sample.session for sample in samples_collection], SESSION
            )
            if labels := [
                sample.label
                for sample in samples_collection
                if sample.label is not None
            ]:
                args[LABEL] = self._merge_field(
                    labels,
                    LABEL,
                )
            args[DATATYPE] = tuple(
                d for sample in samples_collection for d in sample.datatype
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
                    field,
                )

            if "check_consistency" in get_args_from(type_.__init__):
                args["check_consistency"] = False

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

    def _merge_field(self, values: Sequence[Any], field_name: str) -> Any:
        """
        Tries to merge any field.
        """
        if all(isinstance(value, tio.Image) for value in values):
            return self._merge_tio(values)
        elif all(isinstance(value, np.ndarray) for value in values):
            return np.stack(values)
        elif all(isinstance(value, torch.Tensor) for value in values):
            return torch.stack(values)

        try:
            values = set(values)
        except TypeError:
            pass
        else:
            values = list(values)

        if len(values) == 1:
            return values.pop()
        else:
            raise TypeError(
                f"MergeBatches can only merge torchio.Image, numpy.ndarray, or torch.Tensor. For '{field_name}', got: {values}"
            )

    def _merge_tio(self, values: Sequence[tio.Image]) -> tio.Image:
        """
        Merges :py:class:`torchio.Images`.
        """
        self._check_spacing(values)

        tensor = torch.cat([image.tensor for image in values], dim=0)
        affine = values[0].affine
        if all(isinstance(value, tio.LabelMap) for value in values):
            image = tio.LabelMap(tensor=tensor, affine=affine)
        else:
            image = tio.ScalarImage(tensor=tensor, affine=affine)

        return image

    @staticmethod
    def _check_spacing(images: Sequence[tio.Image]) -> None:
        """
        Check if spacing is consistent before concatenating.
        """
        ref_spacing = images[0].spacing
        for image in images[1:]:
            if not np.isclose(ref_spacing, image.spacing, rtol=SPACING_RTOL).all():
                raise RuntimeError(
                    "Trying to concatenate images with different voxel spacing!"
                )
