from typing import Optional, Sequence

import torch
from monai.inferers import SliceInferer
from pydantic import PositiveInt

from clinicadl.transforms.extraction.slice import SliceDirection
from clinicadl.transforms.types import TransformOrConfig
from clinicadl.utils.dictionary.words import IMAGE, OUTPUT
from clinicadl.utils.objects import HasConfig

from .utils import Batched3DTo3DInferer, Batched3DTo3DInfererConfig, ImageOutputType


class SlicesToImageInfererConfig(Batched3DTo3DInfererConfig):
    """Config class for ``SlicesToImageInferer``."""

    slice_direction: SliceDirection
    batch_size: PositiveInt

    @classmethod
    def _get_class(cls):
        return SlicesToImageInferer


class SlicesToImageInferer(Batched3DTo3DInferer, HasConfig[SlicesToImageInfererConfig]):
    """
    Splits a 3D volume into 2D slices, passes them in a 2D neural network, and merges
    the outputs in a 3D output volume.

    See :py:class:`clinicadl.infer.Inferer` and :py:class:`clinicadl.infer.SimpleInferer`
    for more details and examples on ``Inferers``.

    Adapted from :py:class:`monai.inferers.SliceInferer`.

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
        in the neural network.

    postprocessing_on_cpu : bool, default=False
        Whether to necessarily apply postprocessing on CPU. If ``False``, postprocessing will
        be applied on the device where are the data and the neural network.

        .. important::
            ``postprocessing_on_cpu=True`` may potentially change the device on which
            are your input data.

    output_name : str, default="output"
        The name the give to the output in the ``DataPoint``.

        .. important::
            If you postprocessing transform comes from :py:class:`clinicadl.transforms.config`,
            do not forget to specify ``include=["<output_name>"]`` to apply the postprocessing to
            the output of the neural network.

    output_type : Optional[OutputType], default="image"
        Determines the data type of the output:

        - if ``"image"``, the output will be converted to a :py:class:`torchio.ScalarImage`;
        - if ``"mask"``, the output will be converted to a :py:class:`torchio.LabeMap`;
        - if ``None``, the output type will be inferred from the label.

    Examples
    --------

    .. code-block::

        import torch
        from clinicadl.infer import SlicesToImageInferer
        from clinicadl.data.structures.examples import ColinDataPoint
        from clinicadl.networks.nn import ConvEncoder

        net = ConvEncoder(spatial_dims=2, in_channels=1, channels=[2, 4], kernel_size=7)
        datapoint = ColinDataPoint()
        inferer = SlicesToImageInferer(slice_direction=1, batch_size=16)

    .. code-block::

        >>> datapoint.image.shape
        (1, 181, 217, 181)
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
        output_name: str = OUTPUT,
        output_type: Optional[ImageOutputType] = IMAGE,
    ):
        super().__init__(
            slice_direction=slice_direction,
            batch_size=batch_size,
            postprocessing=postprocessing,
            postprocessing_on_cpu=postprocessing_on_cpu,
            output_name=output_name,
            output_type=output_type,
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

        return inferer(inputs=tensor, network=network, **kwargs)
