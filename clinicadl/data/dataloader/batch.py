from typing import Any, List, Union

import torch
import torchio as tio

from clinicadl.data.structures import DataPoint


class SimpleBatch(list[DataPoint]):
    """
    A batch container for :class:`~clinicadl.data.structures.DataPoint` objects.

    This class inherits from the built-in :class:`list` and is specifically designed
    to handle batches of `DataPoint` instances, providing utility methods to
    retrieve their associated image tensors and labels.

    Parameters
    ----------
    samples : list[DataPoint]
        List of :class:`~clinicadl.data.structures.DataPoint` forming the batch.

    Raises
    ------
    ValueError
        If the input list of samples is empty.

    """

    def __init__(self, samples: list[DataPoint]):
        super().__init__(samples)

        if len(self) == 0:
            raise ValueError("The batch is empty")

    def get_images(self) -> Union[torch.Tensor, list[torch.Tensor]]:
        """
        Gathers the images in the batch as :py:class:`torch.Tensor`.

        Returns
        -------
        Union[torch.Tensor, list[torch.Tensor]]
            A tensor containing all the images from the batch if they
            have the same size. A list of tensors otherwise.

            If a tensor is returned, the first dimension is the batch
            dimension.
        """
        images = [sample.image.tensor for sample in self]
        try:
            return torch.stack(images, dim=0)
        except RuntimeError:  # not the same shape
            return images

    def get_labels(self) -> Union[torch.Tensor, List[Any]]:
        """
        Gathers the labels in the batch.

        Returns
        -------
        Union[torch.Tensor, List[Any]]
            A :py:class:`torch.Tensor` or a list containing all the labels from the batch.
            It will be a list if the labels are heterogeneous (e.g. a mask and a scalar) or if any
            of the label is ``None``. Otherwise, it will be a tensor.
        """
        labels = [
            sample.label.tensor
            if isinstance(sample.label, tio.LabelMap)
            else self._dict_to_tensor(sample.label)
            if isinstance(sample.label, dict)
            else sample.label
            for sample in self
        ]

        if all(isinstance(label, torch.Tensor) for label in labels):
            try:
                return torch.stack(labels, dim=0)
            except RuntimeError:  # not the same shape
                return labels

        try:
            return torch.tensor(labels)
        except (TypeError, ValueError, RuntimeError):  # e.g. None in labels
            return labels

    @staticmethod
    def _dict_to_tensor(dict_: dict[str, float]) -> torch.Tensor:
        """
        To convert multi-scalars label.
        """
        return torch.tensor([value for _, value in dict_.items()], dtype=torch.float32)


Batch = Union[SimpleBatch, tuple[SimpleBatch, ...]]


def simple_collate_fn(batch: list[DataPoint]) -> SimpleBatch:
    """For datasets that returns a single Sample."""
    return SimpleBatch(batch)


def tuple_collate_fn(batch: list[tuple[DataPoint, ...]]) -> tuple[SimpleBatch, ...]:
    """For datasets that returns a tuple of Samples."""
    return tuple(SimpleBatch(data) for data in zip(*batch))
