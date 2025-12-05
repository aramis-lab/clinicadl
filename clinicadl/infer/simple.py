from typing import Any, Sequence, TypeVar, Union

import torch
import torch.nn as nn

from clinicadl.data.dataloader import Batch
from clinicadl.data.structures import DataPoint
from clinicadl.transforms.handlers import Postprocessing
from clinicadl.transforms.types import TransformOrConfig
from clinicadl.utils.dictionary.words import IMAGE, OUTPUT

from .base import Inferer

T = TypeVar("T", DataPoint, Batch)


class SimpleInferer(Inferer):
    """
    For classical inference, i.e. when the whole images are passed
    in the neural network and the raw outputs are returned (with a potential
    postprocessing).
    """

    def __init__(self, postprocessing: Sequence[TransformOrConfig]):
        if not postprocessing:
            postprocessing = []
        self.postprocessing = Postprocessing(postprocessing)

    def __call__(
        self, x: Union[DataPoint, Batch], network: nn.Module, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        """
        Simple pass forward in the neural network.

        Parameters
        ----------
        x : Union[DataPoint, Batch]
            The input image(s). Can be a single 3D image in a :py:class:`~clinicadl.data.structures.DataPoint`
            or a :py:class:`~clinicadl.data.dataloader.Batch` of images.
        network : nn.Module
            The neural network.
        args : Any
            Optional args to be passed to ``network``.
        kwargs : Any
            Optional keyword args to be passed to ``network``.

        Returns
        -------
        torch.Tensor
            The raw output of the neural network.
        """
        tensor = self._get_input_tensor(x)

        output = network(tensor, *args, **kwargs)

        x = self._add_output(x, output)

        if isinstance(x, Sample):
            x["output"] = output
            return self.postprocessing.apply(input)
        elif isinstance(x, Batch):
            x.add_field("output", output)
            return self.postprocessing.batch_apply(input)

    @staticmethod
    def _get_input_tensor(x: Union[DataPoint, Batch]) -> torch.Tensor:
        """
        Gets the image(s) and returns a :py:class:`torch.Tensor`.
        """
        if isinstance(x, DataPoint):
            tensor = x.image.tensor
        elif isinstance(x, Batch):
            tensor = x.get_field(IMAGE, torch.float32)
        else:
            raise TypeError(f"'x' can be either a DataPoint or a Batch. Got: {x}")

        return tensor

    @staticmethod
    def _add_output(x: T, output: torch.Tensor) -> T:
        """
        Adds the inference output in the origin data structure.
        """
        if isinstance(x, DataPoint):
            x[OUTPUT] = tio.Sc
        elif isinstance(x, Batch):
            x.add_field(OUTPUT, output)

        return x

    @staticmethod
    def _add_scalar_output(x: T, output: torch.Tensor) -> T:
        """
        Adds the inference output in the origin data structure.
        """
        if isinstance(x, DataPoint):
            x[OUTPUT] = output
        elif isinstance(x, Batch):
            x.add_field(OUTPUT, output)

        return x

    def _postprocess(x: T) -> T:
        """
        Gets the image(s) and returns a :py:class:`torch.Tensor`.
        """
        if isinstance(x, DataPoint):
            return self.postprocessing.apply(x)
        elif isinstance(x, Batch):
            x.add_field("output", output)
            return self.postprocessing.batch_apply(input)
