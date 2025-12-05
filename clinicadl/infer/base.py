from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, Union, overload

import torch.nn as nn

from clinicadl.data.dataloader import Batch
from clinicadl.data.structures import DataPoint
from clinicadl.utils.objects import JsonReaderWriter


class Inferer(JsonReaderWriter, ABC):
    """
    Abstract class for ``Inferers``, which define how an image is passed in a neural network during
    inference.

    The only method to override is :py:meth:`__call__`.

    See Also
    --------
    clinicadl.infer.PatchesToImage
        To feed 3D patches into the neural network and merge the outputs in a 3D image.
    clinicadl.infer.SlicesToImage
        To feed 2D slices into the neural network and merge the outputs in a 3D image.
    clinicadl.infer.PatchesToScalars
        To feed 3D patches into the neural network and fuse the resulting scalar outputs.
    clinicadl.infer.SlicesToScalars
        To feed 2D slices into the neural network and fuse the resulting scalar outputs.
    """

    @overload
    def __call__(
        self,
        x: Union[DataPoint, Sequence[DataPoint]],
        network: nn.Module,
        *args: Any,
        **kwargs: Any,
    ) -> DataPoint:
        ...

    @overload
    def __call__(
        self,
        x: Union[Batch, Sequence[Batch]],
        network: nn.Module,
        *args: Any,
        **kwargs: Any,
    ) -> Batch:
        ...

    @abstractmethod
    def __call__(
        self,
        x: Union[DataPoint, Sequence[DataPoint], Batch, Sequence[Batch]],
        network: nn.Module,
        *args: Any,
        **kwargs: Any,
    ) -> Union[DataPoint, Batch]:
        """
        Defines the inference logic.

        If the input is a :py:class:`~clinicadl.data.structures.DataPoint` or a sequence of ``DataPoint``
        (e.g. the output of a :py:class:`~clinicadl.data.datasets.PairedDataset`), a unique ``DataPoint``
        must be returned.

        If the input is a :py:class:`~clinicadl.data.dataloader.Batch` or a sequence of ``Batch``,
        a unique ``Batch`` must be returned.

        Parameters
        ----------
        x : Union[DataPoint, Sequence[DataPoint], Batch, Sequence[Batch]]
            The input image(s). Can be a :py:class:`~clinicadl.data.structures.DataPoint`
            a :py:class:`~clinicadl.data.dataloader.Batch` of images, or a sequence of either.
        network : nn.Module
            The neural network.
        args : Any
            Optional args to be passed to ``network``.
        kwargs : Any
            Optional keyword args to be passed to ``network``.

        Returns
        -------
        Union[DataPoint, Batch]
            A data structure containing the inference output. If ``DataPoint(s)`` were passed, a unique
            output ``DataPoint`` is returned; if ``Batches(s)`` were passed, a unique ``Batch`` is returned.
        """
