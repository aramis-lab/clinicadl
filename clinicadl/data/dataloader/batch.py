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

    ``Batch`` is simply a list of ``DataPoints``, with additional useful functions.

    Parameters
    ----------
    datapoints : list[DataPoint]
        List of :py:class:`DataPoints <clinicadl.data.structures.DataPoint>` forming the batch.

    Raises
    ------
    ValueError
        If the input list is empty.

    """

    _device: Optional[torch.device] = None
    _non_blocking: bool = False

    def __init__(self, datapoints: list[DataPoint]):
        super().__init__(datapoints)

        if len(self) == 0:
            raise ValueError("The batch is empty!")

    @property
    def device(self) -> torch.device:
        """The device on which the :py:class:`Tensors <torch.Tensor>` in the batch are."""
        return self._device

    def to(
        self, device: Union[str, int, torch.device], non_blocking: bool = False
    ) -> Batch:
        """
        Returns a copy of the ``Batch``, where :py:class:`Tensors <torch.Tensor>` are on the specified device.

        Parameters
        ----------
        device : Union[str, int, torch.device]
            The device where to send the ``Batch``. Can be:

            - an ``int``: the device id;
            - ``"cuda"``;
            - ``"cpu"``
            - ``"cuda-<id>"``: where ``<id>`` is the device id;
            - a :py:class:`torch.device`.

        non_blocking : bool, default=False
            "When non_blocking is set to ``True``, the function attempts to perform the
            conversion asynchronously with respect to the host, if possible.
            This asynchronous behavior applies to both pinned and pageable memory."
            (see :torch:`PyTorch documentation <generated/torch.Tensor.to.html>`).

        Returns
        -------
        Batch
            The copy of the input batch, on the specified device.
        """
        if isinstance(device, str) and not (
            re.match(r"^cuda:.*", device) or device == "cuda" or device == "cpu"
        ):
            raise ValueError(
                "If 'device' is a str, it must be 'cuda' or 'cuda:<device-id>'."
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
        Gathers all the values of a field that is in the ``DataPoints`` of the batch.

        The function will try to return the output as a batch-first :py:class:`torch.Tensor`. If not possible,
        it will return the list of the values.

        If the output is a ``Tensor``, it will be returned on the device passed via :py:meth:`to`, and the desired data type
        as well as the memory format can be specified via ``dtype`` and ``channels_last`` respectively.

        Parameters
        ----------
        field_name : str
            The key to the field in the underlying :py:class:`DataPoints <clinicadl.data.structures.DataPoint>`.
        dtype : Optional[torch.dtype], default=None
            Specifies the output data type, if the output is a ``Tensor``. If ``None``, the output will not
            be cast into a specific data type.
        channels_last : Optional[bool], default=None
            Whether to use `Channels Last Memory Format <https://docs.pytorch.org/tutorials/intermediate/memory_format_tutorial.html>`_
            for the output ``Tensor``. If ``None``, memory format will not be changed.
        ensure_channel_dim : bool, default=False
            If ``True``, a 1D ``Tensor`` output batch (B) will be unsqueezed to a 2D ``Tensor`` with a channel dimension (BC).

        Returns
        -------
        Union[torch.Tensor, list[Any]]
            A :py:class:`torch.Tensor` or a list containing all the values of ``field_name`` in the batch.

        Raises
        ------
        KeyError
            If not all the :py:class:`DataPoints <clinicadl.data.structures.DataPoint>` have the requested ``field_name``.
        ValueError
            If ``channels_last=True`` but the batch tensor is not 4D (BCHW) or 5D (BCDHW).

        Examples
        --------
        .. code-block:: python

            from clinicadl.data.structures import ColinDataPoint
            from clinicadl.data.dataloader import Batch
            datapoint = ColinDataPoint()
            batch = Batch([datapoint, datapoint])

        .. code-block:: python

            >>> datapoint
            ColinDataPoint(Keys: ('image', 'label', 'participant', 'session', 'head'); images: 3)
            >>> datapoint["label"]
            LabelMap(shape: (1, 181, 217, 181); spacing: (1.00, 1.00, 1.00); orientation: RAS+; dtype: torch.ShortTensor; memory: 13.6 MiB)
            >>> datapoint["participant"]
            'sub-colin'

        .. code-block:: python

            >>> batch.get_field("label").shape
            torch.Size([2, 1, 181, 217, 181])
            >>> batch.get_field("participant")
            ['sub-colin', 'sub-colin']

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

    def get_field(
        self, field_name: str, ensure_channel_dim: bool = False
    ) -> Union[torch.Tensor, list[Any]]:
        """
        Gathers all the values of a field in the batch.

        Parameters
        ----------
        field_name : str
            The key to the field in the underlying :py:class:`DataPoints <clinicadl.data.structures.DataPoint>`.

        Returns
        -------
        Union[torch.Tensor, list[Any]]
            A :py:class:`torch.Tensor` or a list containing all the values of ``field_name`` in the batch.
            It will be a list a :py:class:`torch.Tensor` if all the values could be converted to tensors,
            and if all tensors have the same shape.
        """
        values = []
        for datapoint in self:
            try:
                value = datapoint[field_name]
            except KeyError as e:
                raise KeyError(
                    f"You want to get '{field_name}', but there is no such key in some DataPoints in the batch."
                ) from e

            try:
                value = self._to_tensor(value)
            except TypeError:
                pass

            values.append(value)

        if all(isinstance(v, torch.Tensor) for v in values):
            try:
                values = torch.stack(values, dim=0)
            except RuntimeError:  # not the same shape
                return values
        else:
            try:
                values = torch.tensor(values)
            except (TypeError, ValueError, RuntimeError):  # e.g. None in labels
                return values

        if len(values.shape) == 1 and ensure_channel_dim:  # at least two dimensions
            return values.unsqueeze(1)

        return values

    @staticmethod
    def _to_tensor(value: Any) -> torch.Tensor:
        """
        Tries to convert to a tensor.
        """
        if isinstance(value, tio.ScalarImage):
            value = value.tensor.float()
        elif isinstance(value, tio.LabelMap):
            value = value.tensor.int()
        elif isinstance(value, np.ndarray):
            value = torch.from_numpy(value)
        elif isinstance(value, numbers.Number):
            value = torch.tensor(value)
        elif isinstance(value, torch.Tensor):
            pass
        else:
            raise TypeError

        return value

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
