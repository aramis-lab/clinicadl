from abc import ABC, abstractmethod
from typing import Any, Optional, Union

import torch
import torch.nn as nn

from clinicadl.data.dataloader import Batch
from clinicadl.data.structures import DataPoint
from clinicadl.utils.dictionary.words import IMAGE
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

    @abstractmethod
    def __call__(
        self,
        x: Union[DataPoint, Batch],
        network: nn.Module,
        input_dtype: Optional[torch.dtype] = None,
        **kwargs: Any,
    ) -> Union[DataPoint, Batch]:
        """
        Defines the inference logic.

        Parameters
        ----------
        x : Union[TDataPoint, Batch]
            The input image(s). Can be a :py:class:`~clinicadl.data.structures.DataPoint` or
            a :py:class:`~clinicadl.data.dataloader.Batch` of images.
        network : nn.Module
            The neural network.
        input_dtype : Optional[torch.dtype], default=None
            The data type to which the input image is converted before being processed by ``network``.
            If ``None``, single precision (i.e. ``float32``) will be used (except if the inferer is run
            in an :term:`AMP` context).
        kwargs : Any
            Optional keyword args to be passed to ``network``.

        Returns
        -------
        Union[TDataPoint, Batch]
            The same data structure as the input, containing the inference output.
        """

    @staticmethod
    def _get_input_tensor(
        x: Union[DataPoint, Batch], input_dtype: Optional[torch.dtype] = None
    ) -> torch.Tensor:
        """
        Gets the image(s) and returns a :py:class:`torch.Tensor`.
        """
        if isinstance(x, DataPoint):
            tensor = x.image.tensor.to(dtype=input_dtype)
        elif isinstance(x, Batch):
            tensor = x.get_field(IMAGE, dtype=input_dtype)
        else:
            raise TypeError(f"'x' can be either a DataPoint or a Batch. Got: {x}")

        return tensor
