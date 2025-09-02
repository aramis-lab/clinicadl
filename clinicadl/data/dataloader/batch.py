from __future__ import annotations

import re
from copy import deepcopy
from typing import Any, Optional, Union

import numpy as np
import torch
import torchio as tio

from clinicadl.data.structures import DataPoint


class Batch(list[DataPoint]):
    """
    A batch container for :class:`~clinicadl.data.structures.DataPoint` objects.

    This class inherits from the built-in :class:`list` and is specifically designed
    to handle batches of ``DataPoint`` instances.

    Parameters
    ----------
    samples : list[DataPoint]
        List of :py:class:`~clinicadl.data.structures.DataPoint` forming the batch.

    Raises
    ------
    ValueError
        If the input list of samples is empty.

    """

    _device: Optional[torch.device] = None
    _non_blocking: bool = False

    def __init__(self, samples: list[DataPoint]):
        super().__init__(samples)

        if len(self) == 0:
            raise ValueError("The batch is empty!")

    @property
    def device(self) -> torch.device:
        """The device on which are the tensors in the batch."""
        return self._device

    def to(
        self, device: Union[str, int, torch.device], non_blocking: bool = False
    ) -> Batch:
        """
        Returns a copy of the ``Batch`` on the specified device.

        Parameters
        ----------
        device : Union[str, int, torch.device]
            The device where to send the ``Batch``.
        non_blocking : bool, default=False
            "When non_blocking is set to ``True``, the function attempts to perform the
            conversion asynchronously with respect to the host, if possible.
            This asynchronous behavior applies to both pinned and pageable memory."
            (see :torch:`PyTorch documentation <generated/torch.Tensor.to.html>`).

        Returns
        -------
        Batch
            The copy of the input batch, on the wanted device.
        """
        if isinstance(device, str) and not re.match(r"^cuda:.*", device):
            raise ValueError(
                "If 'device' is a str, it must be like 'cuda:<device-id>'."
            )

        batch = deepcopy(self)

        for datapoint in batch:
            for name, value in datapoint.items():
                if isinstance(value, torch.Tensor):
                    datapoint[name] = value.to(device, non_blocking=non_blocking)
            datapoint.update_attributes()

        batch._device = torch.device(device)
        batch._non_blocking = non_blocking

        return batch

    def get_field(
        self,
        field_name: str,
        dtype: Optional[torch.dtype] = None,
        channels_last: Optional[bool] = None,
        ensure_channel_dim: bool = False,
    ) -> Union[torch.Tensor, list[Any]]:
        """
        Gathers all the values of a field in the :py:class:`~clinicadl.data.structures.DataPoint` in the batch.

        The function will try to return the output as a batch-first :py:class`torch.Tensor`. If not possible,
        it will return the list of the values.

        The tensor will be returned on the device passed via :py:meth:`to`.

        Besides, if the output is a ``Tensor``, the desired data type and the memory format can be specified via
        ``dtype`` and ``channels_last`` respectively.

        Parameters
        ----------
        field_name : str
            The key to the field in the underlying :py:class:`DataPoints <clinicadl.data.structures.DataPoint>`.
        dtype : Optional[torch.dtype], default=None
            Specifies the output data type, if the output is a ``Tensor``. If ``None``, the output will not
            be cast into a specific data type.
        channels_last : Optional[bool], default=None
            Whether to use `Channels Last Memory Format <https://docs.pytorch.org/tutorials/intermediate/memory_format_tutorial.html>`_
            for the output tensor.
        ensure_channel_dim : bool, default=False
            If ``True``, a 1D tensor output batche (B) will be unsqueezed to 2D tensor with a channel dimension (BC).

        Returns
        -------
        Union[torch.Tensor, list[Any]]
            A :py:class:`torch.Tensor` or a list containing all the values of ``field_name`` in the batch.

        Raises
        ------
        KeyError
            If not all the :py:class:`~clinicadl.data.structures.DataPoint` have the requested ``field_name``.
        ValueError
            If ``channels_last=True`` but the batch tensor is not 4D (BCHW) or 5D (BCDHW).
        """
        # collect all the values and try to convert them to tensors
        batch = []
        try:
            for datapoint in self:
                value = self._get_field(datapoint, field_name)

                try:
                    value = self._to_tensor(value)
                except TypeError:
                    raise StopIteration

                batch.append(value)

        except StopIteration:  # some field values cannot be converted to tensors
            return [self._get_field(datapoint, field_name) for datapoint in self]
        else:  # now let's merge in one tensor
            try:
                batch = torch.stack(batch, dim=0)
            except RuntimeError:  # not the same shape, batch as tensor is not possible
                return [self._get_field(datapoint, field_name) for datapoint in self]

        # format the batch tensor
        if len(batch.shape) == 1 and ensure_channel_dim:  # at least two dimensions
            batch = batch.unsqueeze(1)

        memory_format = self._get_memory_format(batch, channels_last=channels_last)

        return batch.to(
            dtype=dtype,
            device=self._device,
            non_blocking=self._non_blocking,
            memory_format=memory_format,
        )

    @staticmethod
    def _get_field(datapoint: DataPoint, field_name: str) -> Any:
        """Returns the specified field."""
        try:
            return datapoint[field_name]
        except KeyError as e:
            raise KeyError(
                f"You want to get '{field_name}', but there is no such key in some DataPoints in the batch."
            ) from e

    @classmethod
    def _to_tensor(cls, value: Any) -> torch.Tensor:
        """
        Tries to convert to a tensor.
        """
        if isinstance(value, tio.ScalarImage):
            return value.tensor.float()
        elif isinstance(value, tio.LabelMap):
            return value.tensor.int()
        elif isinstance(value, np.ndarray):
            return torch.from_numpy(value)
        elif isinstance(value, dict):
            return cls._to_tensor(list(value.values()))
        elif isinstance(value, torch.Tensor):
            return value
        else:
            try:
                return torch.tensor(value)
            except (TypeError, ValueError, RuntimeError) as exc:
                raise TypeError from exc

    @staticmethod
    def _get_memory_format(
        tensor: torch.Tensor,
        channels_last: Optional[bool],
    ) -> torch.memory_format:
        """
        Gets the desired memory format.
        """
        if channels_last:
            if len(tensor.shape) == 4:
                return torch.channels_last
            elif len(tensor.shape) == 5:
                return torch.channels_last_3d
            else:
                raise ValueError(
                    "To use Channels Last memory format, the output tensor must be 4D (BCHW) or 5D (BCDHW). "
                    f"Here it is {int(len(tensor.shape))}D."
                )
        elif channels_last is False:
            return torch.contiguous_format
        else:
            return torch.preserve_format


BatchType = Union[Batch, tuple[Batch, ...]]


def simple_collate_fn(batch: list[DataPoint]) -> Batch:
    """For datasets that returns a single Sample."""
    return Batch(batch)


def tuple_collate_fn(batch: list[tuple[DataPoint, ...]]) -> tuple[Batch, ...]:
    """For datasets that returns a tuple of Samples."""
    return tuple(Batch(data) for data in zip(*batch))
