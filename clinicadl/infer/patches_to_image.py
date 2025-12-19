from collections.abc import Sequence
from enum import Enum
from typing import Any, Optional, Union

import torch
from monai.inferers import SlidingWindowInferer
from pydantic import (
    NonNegativeFloat,
    PositiveFloat,
    PositiveInt,
    field_validator,
)

from clinicadl.transforms.types import TransformOrConfig
from clinicadl.utils.dictionary.words import IMAGE, OUTPUT
from clinicadl.utils.objects import HasConfig

from .utils import Batched3DTo3DInferer, Batched3DTo3DInfererConfig, ImageOutputType


class AveragingMode(str, Enum):
    """Averaging method where patches overlap."""

    CONSTANT = "constant"
    GAUSSIAN = "gaussian"


class PatchesToImageInfererConfig(Batched3DTo3DInfererConfig):
    """Config class for ``PatchesToImageInferer``."""

    patch_size: tuple[PositiveInt, PositiveInt, PositiveInt]
    overlap: tuple[NonNegativeFloat, NonNegativeFloat, NonNegativeFloat]
    avg_mode: AveragingMode
    sigma_scale: PositiveFloat
    batch_size: PositiveInt

    @field_validator("patch_size", "overlap", mode="before")
    @classmethod
    def _ensure_tuple(
        cls,
        value: Any,
    ) -> tuple:
        """
        Ensures that arguments is a tuple.
        """
        if not isinstance(value, Sequence):
            return (value, value, value)
        return value

    @field_validator("overlap", mode="after")
    @classmethod
    def _overlap_validator(cls, value: tuple) -> tuple:
        """Checks that overlap is between 0 and 1 if it is a float."""
        for v in value:
            assert (
                0 <= v < 1
            ), f"If 'overlap' is a float, it must be between 0 (included) and 1 (excluded). Got {v}"
        return value

    @classmethod
    def _get_class(cls):
        return PatchesToImageInferer


class PatchesToImageInferer(
    Batched3DTo3DInferer, HasConfig[PatchesToImageInfererConfig]
):
    """
    Splits a 3D volume into 3D patches, passes them in a 3D neural network, and merges
    the outputs in a 3D output volume.

    See :py:class:`clinicadl.infer.Inferer` and :py:class:`clinicadl.infer.SimpleInferer`
    for more details and examples on ``Inferers``.

    Adapted from :py:class:`monai.inferers.SlidingWindowInferer`.

    Parameters
    ----------
    patch_size : Union[int, tuple[int, int, int]]
        The size of the patches. If a single value is passed, the same patch size will be used for the three
        spatial dimensions.
    overlap: Union[float, tuple[float, float, float]], default=0.0
        A ``float`` in :math:`[0.0, 1.0)` that defines relative patch overlap in each dimension.
        If a single value is passed, the same overlap will be used for the three spatial dimensions.
    avg_mode : AveragingMode, default="constant"
        How to average the outputs when patches overlap.

        - ``"constant"``: gives equal weight to all predictions;
        - ``"gaussian"``: gives less weight to the prediction on edges of patches.

    sigma_scale: float, default=0.125,
        The standard deviation coefficient of the Gaussian window when ``avg_mode`` is ``"gaussian"``.
        The actual sigma is ``sigma_scale * patch_size``.
    batch_size : int, default=1
        The size of the batch passed to the neural network. If you pass a batch of images to
        the inferer, this batch will be rearranged to match ``batch_size``.

        E.g. if a batch of :math:`2` images is passed, with `3` patches in each image, and ``batch_size=4``, then
        the first batch passed to the neural network will contain the three patches of the first image,
        and the first patch of the second.

    postprocessing : Optional[Sequence[TransformOrConfig]], default=None
        To apply postprocessing transformations (e.g. activations) after the pass forward
        in the neural network and output fusion.

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

        net = AutoEncoder(in_shape=(1, 64, 64, 64), latent_size=16, conv_args={"channels": [2]})
        datapoint = ColinDataPoint()
        inferer = PatchesToImageInferer(patch_size=64, batch_size=16, overlap=1/5)


    .. code-block::

        >>> datapoint.image.shape
        (1, 181, 217, 181)
        >>> with torch.no_grad(): out = inferer(datapoint, net)
        >>> out["output"].shape
        (1, 181, 217, 181)

    See Also
    --------
    clinicadl.infer.SimpleInferer
        For classical inference.
    """

    config: PatchesToImageInfererConfig
    _config_type = PatchesToImageInfererConfig

    def __init__(
        self,
        patch_size: Union[int, tuple[PositiveInt, PositiveInt, int]],
        overlap: Union[float, tuple[float, float, float]] = 0.25,
        avg_mode: AveragingMode = AveragingMode.CONSTANT,
        sigma_scale: float = 0.125,
        batch_size: int = 1,
        postprocessing: Optional[Sequence[TransformOrConfig]] = None,
        postprocessing_on_cpu: bool = False,
        output_name: str = OUTPUT,
        output_type: Optional[ImageOutputType] = IMAGE,
    ):
        super().__init__(
            patch_size=patch_size,
            overlap=overlap,
            avg_mode=avg_mode,
            sigma_scale=sigma_scale,
            batch_size=batch_size,
            postprocessing=postprocessing,
            postprocessing_on_cpu=postprocessing_on_cpu,
            output_name=output_name,
            output_type=output_type,
        )
        self._sliding_window = SlidingWindowInferer(
            roi_size=self.config.patch_size,
            sw_batch_size=self.config.batch_size,
            overlap=self.config.overlap,
            sw_device=None,  # where inference is done
            device=torch.device("cpu")  # where patch are assembled
            if postprocessing_on_cpu
            else None,
            mode=self.config.avg_mode,
            sigma_scale=self.config.sigma_scale,
        )

    def _forward_pass(
        self, tensor: torch.Tensor, network: torch.nn.Module, **kwargs
    ) -> torch.Tensor:
        self._check_shape(tensor)

        return self._sliding_window(inputs=tensor, network=network, **kwargs)

    def _check_shape(self, tensor: torch.Tensor) -> None:
        """
        Checks spatial shape of the input image(s).
        """
        spatial_shape = tensor.shape[-3:]
        if spatial_shape < self.config.patch_size:
            raise ValueError(
                f"'patch_size' is bigger than the image. Got an image of spatial shape {spatial_shape} but patch_size={self.config.patch_size}"
            )
