from typing import Optional, Sequence, Union

import torch
import torchio as tio
from monai.inferers import SliceInferer
from pydantic import PositiveInt

from clinicadl.data.dataloader import Batch
from clinicadl.data.structures import DataPoint
from clinicadl.transforms.extraction.slice import SliceDirection
from clinicadl.transforms.types import TransformOrConfig
from clinicadl.utils.dictionary.words import IMAGE
from clinicadl.utils.objects import HasConfig

from .base import BaseInferer, BaseInfererConfig


class SlicesToImageInfererConfig(BaseInfererConfig):
    """Config class for ``SlicesToImageInferer``."""

    slice_direction: SliceDirection
    batch_size: PositiveInt

    @classmethod
    def _get_class(cls):
        return SlicesToImageInferer


class SlicesToImageInferer(BaseInferer, HasConfig[SlicesToImageInfererConfig]):
    """
    Splits a 3D volume into 2D slices, passes them in a 2D neural network, and merges
    the outputs in a 3D output volume.

    See :py:class:`clinicadl.infer.Inferer` and :py:class:`clinicadl.infer.SimpleInferer`
    for more details and examples on ``Inferers``.

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

    postprocessing : Optional[Sequence[TransformOrConfig]], default=None
        To apply postprocessing transformations (e.g. activations) after the pass forward
        in the neural network and output fusion.

        .. important::
            If you postprocessing transform comes from :py:class:`clinicadl.transforms.config`,
            do not forget to specify ``include=["output"]`` to apply the postprocessing to
            the output of the neural network.

    postprocessing_on_cpu : bool, default=False
        Whether to necessarily apply postprocessing on CPU. If ``False``, postprocessing will
        be applied on the device where are the data and the neural network.

        .. important::
            ``postprocessing_on_cpu=True`` may potentially change the device on which
            are your input data.

    Examples
    --------

    .. code-block::

        import torch
        from clinicadl.infer import SlicesToImageInferer
        from clinicadl.data.structures.examples import ColinDataPoint
        from clinicadl.networks.nn import ConvEncoder

        net = ConvEncoder(spatial_dims=2, in_channels=1, channels=[2, 4], kernel_size=7)
        datapoint = ColinDataPoint()

    .. code-block::

        >>> datapoint.image.shape
        (1, 181, 217, 181)
        >>> inferer = SlicesToImageInferer(slice_direction=1, batch_size=16)
        >>> with torch.no_grad(): out = inferer(datapoint, net)
        >>> out["output"].shape
        (4, 169, 217, 169)  # 2D neural network applies to the 217 coronal slices

    See Also
    --------
    clinicadl.infer.SimpleInferer
        For classical inference.
    """

    config: SlicesToImageInfererConfig
    _config_type = SlicesToImageInfererConfig

    def __init__(
        self,
        slice_direction: SliceDirection,
        batch_size: int = 1,
        postprocessing: Optional[Sequence[TransformOrConfig]] = None,
        postprocessing_on_cpu: bool = False,
    ):
        super().__init__(
            slice_direction=slice_direction,
            batch_size=batch_size,
            postprocessing=postprocessing,
            postprocessing_on_cpu=postprocessing_on_cpu,
        )

    def _forward_pass(
        self, tensor: torch.Tensor, network: torch.nn.Module, **kwargs
    ) -> torch.Tensor:
        shape_2d = list(tensor.shape[2:])
        shape_2d.pop(self.config.slice_direction)

        inferer = SliceInferer(
            spatial_dim=self.config.slice_direction,
            roi_size=shape_2d,  # we always keep the whole slice here
            sw_batch_size=self.config.batch_size,
        )

        return inferer(inputs=tensor, network=network)

    @classmethod
    def _get_input_tensor(
        cls, x: Union[DataPoint, Batch], input_dtype: Optional[torch.dtype] = None
    ) -> torch.Tensor:
        tensor = super()._get_input_tensor(x, input_dtype)

        if isinstance(x, DataPoint):
            tensor = x.image.tensor.to(dtype=input_dtype)
            assert (
                len(tensor.shape) == 4
            ), f"{cls.__name__} only accepts 4D images (including 1 channel dimension). Got shape: {tensor.shape}"
            return tensor.unsqueeze(0)  # SliceInferer only accepts batched outputs

        elif isinstance(x, Batch):
            tensor = x.get_field(IMAGE, dtype=input_dtype)
            assert (
                len(tensor.shape) == 5
            ), f"{cls.__name__} only accepts 4D images (including 1 channel dimension). Got a batch of images with shape: {tensor.shape[1:]}"

        return tensor

    @classmethod
    def _format_output(
        cls, x: DataPoint, output: torch.Tensor
    ) -> Union[tio.Image, torch.Tensor]:
        if isinstance(x, DataPoint):
            output = output.squeeze(0)

        return super()._format_output(x, output)
