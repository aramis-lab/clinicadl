from typing import Optional, Union

import torch

from clinicadl.data.dataloader import Batch
from clinicadl.data.structures import DataPoint

from .base import BaseInferer


class Batched3DTo3DInferer(BaseInferer):
    """
    Base class for inferers that work only with **batches of 4D images**,
    and that return batches of 4D images.
    """

    @classmethod
    def _get_input_tensor(
        cls, x: Union[DataPoint, Batch], input_dtype: Optional[torch.dtype] = None
    ) -> torch.Tensor:
        tensor = super()._get_input_tensor(x, input_dtype)

        if isinstance(x, DataPoint):
            assert (
                len(tensor.shape) == 4
            ), f"{cls.__name__} only accepts 4D images (including 1 channel dimension). Got shape: {tensor.shape}"
            return tensor.unsqueeze(0)  # only accepts batched outputs

        elif isinstance(x, Batch):
            assert (
                len(tensor.shape) == 5
            ), f"{cls.__name__} only accepts 4D images (including 1 channel dimension). Got a batch of images with shape: {tensor.shape[1:]}"

        return tensor

    def _add_output(self, x: Union[DataPoint, Batch], output: torch.Tensor) -> None:
        if isinstance(x, DataPoint):
            output = output.squeeze(0)  # remove batch dimension

        super()._add_output(x, output)
