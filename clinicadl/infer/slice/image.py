from typing import Any

import torch
from monai.inferers import SliceInferer as MonaiSliceInferer

from clinicadl.transforms.extraction.slice import SliceDirection

from ..base import Inferer


class SliceImageInferer(Inferer):
    """
    Splits a 3D volume into 2D slices, passes them in a 2D neural network, and merges
    the outputs in a 3D output volume.

    Parameters
    ----------
    slice_direction : SliceDirection, default=0
        The slicing direction. Can be ``0`` (sagittal direction), ``1`` (coronal) or ``2`` (axial).
    batch_size : int, default=1
        The size of the batch passed to the neural network. If you pass a batch of images to
        the inferer, this batch will be rearranged to match ``batch_size``.

        E.g. if a batch of :math:`2` images is passed, with `3` slices in each image, and ``batch_size=4``, then
        the first batch passed to the neural network will contain the three slices of the first image,
        and the first slice of the second.
    """

    def __init__(self, slice_direction: SliceDirection, batch_size: int = 1):
        self.slice_direction = SliceDirection(slice_direction).value
        self.batch_size = batch_size

    def __call__(
        self, x: torch.Tensor, network: torch.nn.Module, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        """
        Passes all the slices in the 2D neural network, and merge the output slices
        in a single image.

        Parameters
        ----------
        x : torch.Tensor
            The input image(s). Can be a single 3D image (CHWD) or a batch (NCHWD).
        network : nn.Module
            The 2D neural network.

        Returns
        -------
        torch.Tensor
            The raw output of the neural network.
        """
        self._check_input(x)
        if len(x.shape) == 4:
            x = x.unsqueeze(0)

        shape_2d = x.shape[2:]
        del shape_2d[self.slice_direction]

        inferer = MonaiSliceInferer(
            spatial_dim=self.slice_direction,
            roi_size=shape_2d,
            sw_batch_size=self.batch_size,
        )  # we always keep the whole slice here so roi_size=shape_2d

        return inferer(inputs=x, network=network)
