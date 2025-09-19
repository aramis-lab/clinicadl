from __future__ import annotations

from copy import deepcopy
from typing import Any, Optional, Sequence, Union

import numpy as np
import torch
import torchio as tio

from clinicadl.data.structures import DataPoint
from clinicadl.utils.device import DeviceType, check_device


class Batch(list[DataPoint]):
    """
    A batch container for :class:`~clinicadl.data.structures.DataPoint` objects.

    ``Batch`` is simply a list of ``DataPoints``, with additional useful functions.

    Parameters
    ----------
    datapoints : Sequence[DataPoint]
        Sequence of :py:class:`DataPoints <clinicadl.data.structures.DataPoint>` forming the batch.

    Raises
    ------
    ValueError
        If the input sequence is empty.

    """

    _device: Optional[torch.device] = None
    _non_blocking: bool = False
    _channels_last: bool = False

    def __init__(self, datapoints: list[DataPoint]):
        super().__init__(datapoints)

        if len(self) == 0:
            raise ValueError("The batch is empty!")

    @property
    def device(self) -> Optional[torch.device]:
        """
        The device on which the :py:class:`Tensors <torch.Tensor>` in the batch are.
        It is also the device on which :py:class:`Tensors <torch.Tensor>` will be returned
        by :py:meth:`get_field`.\n
        It is specified with :py:meth:`to`. If ``device=None``, it means that :py:meth:`to` has not
        been called, and that the ``Tensors`` inside the batch can be on any device.
        """
        return self._device

    @property
    def channels_last(self) -> bool:
        """
        Whether `Channels Last Memory Format <https://docs.pytorch.org/tutorials/intermediate/memory_format_tutorial.html>`_
        is used for 4D (NCWH) or 5D (NCDWH) :py:class:`Tensors <torch.Tensor>` returned by :py:meth:`get_field`.
        It is specified via :py:meth:`to`.\n
        By default, it is ``False``.
        """
        return self._channels_last

    def to(
        self,
        device: Optional[DeviceType] = None,
        non_blocking: bool = False,
        channels_last: Optional[bool] = None,
    ) -> Batch:
        """
        Returns a copy of the ``Batch``, where :py:class:`Tensors <torch.Tensor>` are on the specified device
        and with the specified memory format.

        .. note::
            If ``device`` and ``channels_last`` are both ``None``, the original ``Batch`` will be returned.

        Parameters
        ----------
        device : Optional[DeviceType], default=None
            The device where to send the ``Batch``. Can be:

            - an ``int``: the device id;
            - ``"cuda"``;
            - ``"cpu"``
            - ``"cuda-<id>"``: where ``<id>`` is the device id;
            - a :py:class:`torch.device`;
            - ``None``: the device won't be changed.

        non_blocking : bool, default=False
            "When ``non_blocking`` is set to ``True``, the function attempts to perform the
            conversion asynchronously with respect to the host, if possible.
            This asynchronous behavior applies to both pinned and pageable memory."
            (see :torch:`PyTorch documentation <generated/torch.Tensor.to.html>`)
        channels_last : Optional[bool], default=None
            Whether to use `Channels Last Memory Format <https://docs.pytorch.org/tutorials/intermediate/memory_format_tutorial.html>`_
            for 4D (NCWH) or 5D (NCDWH) :py:class:`Tensors <torch.Tensor>` returned by :py:meth:`get_field`.\n
            If ``False``, the default contiguous memory format will be used.\n
            If ``None``, the current memory format will be kept.

        Returns
        -------
        Batch
            The copy of the input batch, on the specified device, and with the specified memory format.

        Examples
        --------
        .. code-block:: python

            from clinicadl.data.structures import ColinDataPoint
            from clinicadl.data.dataloader import Batch
            import torch

            datapoint = ColinDataPoint()
            datapoint["tensor"] = torch.tensor([1])
            batch = Batch([datapoint, datapoint])
            batch_gpu = batch.to("cuda", non_blocking=True, channels_last=True)

        .. code-block:: python

            >>> batch_gpu.device
            device(type='cuda')
            >>> batch_gpu[0]["tensor"].device   # all the tensors inside are affected
            device(type='cuda')
            >>> batch_gpu.get_field("image").device   # get_field is affected
            device(type='cuda')

        .. code-block:: python

            >>> batch.get_field("image").stride()   # contiguous memory format (default)
            (7109137, 7109137, 39277, 181, 1)
            >>> batch_gpu.get_field("image").stride()   # Channels-last memory format
            (7109137, 1, 39277, 181, 1)

        """
        if device is None and channels_last is None:
            return self

        batch = deepcopy(self)

        if device is not None:
            batch._device = check_device(device)
            batch._non_blocking = non_blocking

            for datapoint in batch:
                for name, value in datapoint.items():
                    if isinstance(value, torch.Tensor):
                        datapoint[name] = value.to(
                            batch.device, non_blocking=non_blocking
                        )
                datapoint.update_attributes()

        if channels_last is not None:
            batch._channels_last = channels_last

        return batch

    def get_field(
        self,
        field_name: str,
        dtype: Optional[torch.dtype] = None,
        ensure_channel_dim: bool = False,
    ) -> Union[torch.Tensor, list[Any]]:
        """
        Gathers all the values of a field that is in the :py:class:`DataPoints <clinicadl.data.structures.DataPoint>`
        of the batch.

        The function will try to return the output as a batch-first :py:class:`~torch.Tensor`. If not possible,
        it will return the list of the values.

        If the output is a ``Tensor``, it will respect the device and the memory format potentially
        specified with :py:meth:`to`. Besides, the desired data type can be specified here via ``dtype``.

        Parameters
        ----------
        field_name : str
            The key to the field in the underlying :py:class:`DataPoints <clinicadl.data.structures.DataPoint>`.
        dtype : Optional[torch.dtype], default=None
            Specifies the output data type, if the output is a :py:class:`~torch.Tensor`. If ``None``, the output will not
            be cast into a specific data type.
        ensure_channel_dim : bool, default=False
            If ``True``, a 1D ``Tensor`` output batch (N) will be unsqueezed to a 2D ``Tensor`` with a channel dimension (NC).

        Returns
        -------
        Union[torch.Tensor, list[Any]]
            A :py:class:`~torch.Tensor` or a list containing all the values of ``field_name`` in the batch.

        Raises
        ------
        KeyError
            If not all the :py:class:`DataPoints <clinicadl.data.structures.DataPoint>` have the requested ``field_name``.

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

                if (
                    hasattr(datapoint, "squeeze")
                    and hasattr(datapoint, "slice_direction")
                    and getattr(datapoint, "squeeze")
                ):
                    squeezed_dim = getattr(datapoint, "slice_direction")
                else:
                    squeezed_dim = None

                try:
                    value = self._to_tensor(value, squeeze_img_dim=squeezed_dim)
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

        memory_format = self._get_memory_format(
            batch, channels_last=self._channels_last
        )

        return batch.to(
            dtype=dtype,
            device=self._device,
            non_blocking=self._non_blocking,
            memory_format=memory_format,
        )

    def add_field(self, field_name: str, values: Sequence[Any]) -> None:
        """
        To add a field to the :py:class:`DataPoints <clinicadl.data.structures.DataPoint>`
        inside the current ``Batch``.

        This method is useful for example when a neural network returns a batch of outputs
        that one wants to store in the original ``Batch``.

        Parameters
        ----------
        field_name : str
            The name fo the field.
        values : Sequence[Any]
            The values of the field for each element of the ``Batch``. Obviously, the sequence must
            be the same size as the ``Batch``.

        Examples
        --------
        .. code-block:: python

            from clinicadl.data.structures import ColinDataPoint
            from clinicadl.data.dataloader import Batch

            datapoint = ColinDataPoint()
            batch = Batch([datapoint, datapoint])

        .. code-block:: python

            >>> batch[0]
            ColinDataPoint(Keys: ('image', 'label', 'participant', 'session', 'head'); images: 3)

        .. code-block:: python

            >>> import torch
            >>> batch.add_field("output", torch.randn(2, 1, 3, 3, 3))
            >>> batch[0]
            ColinDataPoint(Keys: ('image', 'label', 'participant', 'session', 'head', 'output'); images: 3)
            >>> batch[0]["output"].shape
            torch.Size([1, 3, 3, 3])

        """
        assert len(values) == len(self), (
            f"'values' must have the same length as the batch. Got {len(values)} values, "
            f"whereas the batch has only {len(self)} elements"
        )
        for datapoint, value in zip(self, values):
            datapoint[field_name] = value
            datapoint.update_attributes()

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
    def _to_tensor(
        cls, value: Any, squeeze_img_dim: Optional[int] = None
    ) -> torch.Tensor:
        """
        Tries to convert to a tensor.
        """
        if isinstance(value, tio.Image):
            tensor = value.tensor.clone()
            if squeeze_img_dim is not None:
                tensor.squeeze_(dim=squeeze_img_dim + 1)

            if isinstance(value, tio.ScalarImage):
                return tensor.float()
            elif isinstance(value, tio.LabelMap):
                return tensor.int()

        elif isinstance(value, np.ndarray):
            return torch.from_numpy(value)
        elif isinstance(value, dict):
            return cls._to_tensor(list(value.values()))
        elif isinstance(value, torch.Tensor):
            return value.clone()
        else:
            try:
                return torch.tensor(value)
            except (TypeError, ValueError, RuntimeError) as exc:
                raise TypeError from exc

    @staticmethod
    def _get_memory_format(
        tensor: torch.Tensor,
        channels_last: bool,
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
            return torch.contiguous_format


BatchType = Union[Batch, tuple[Batch, ...]]


def simple_collate_fn(batch: list[DataPoint]) -> Batch:
    """For datasets that returns a single Sample."""
    return Batch(batch)


def tuple_collate_fn(batch: list[tuple[DataPoint, ...]]) -> tuple[Batch, ...]:
    """For datasets that returns a tuple of Samples."""
    return tuple(Batch(data) for data in zip(*batch))
