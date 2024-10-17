from enum import Enum


class ImplementedNetworks(str, Enum):
    """Implemented neural networks in ClinicaDL."""

    MLP = "MLP"
    CONV_ENCODER = "ConvEncoder"
    CONV_DECODER = "ConvDecoder"
    CNN = "CNN"
    GENERATOR = "Generator"
    AE = "AutoEncoder"
    VAE = "VAE"
    DENSENET = "DenseNet"
    DENSENET_121 = "DenseNet-121"
    DENSENET_161 = "DenseNet-161"
    DENSENET_169 = "DenseNet-169"
    DENSENET_201 = "DenseNet-201"
    RESNET = "VarFullyConnectedNet"
    RESNET_18 = "ResNet-18"
    RESNET_34 = "ResNet-34"
    RESNET_50 = "ResNet-50"
    RESNET_101 = "ResNet-101"
    RESNET_152 = "ResNet-152"
    SE_RESNET = "SE-ResNet"
    SE_RESNET_50 = "SE-ResNet-50"
    SE_RESNET_101 = "SE-ResNet-101"
    SE_RESNET_152 = "SE-ResNet-152"
    UNET = "UNet"
    ATT_UNET = "AttentionUNet"
    VIT = "ViT"
    VIT_B_16 = "ViT-B/16"
    VIT_B_32 = "ViT-B/32"
    VIT_L_16 = "ViT-L/16"
    VIT_L_32 = "ViT-L/32"

    @classmethod
    def _missing_(cls, value):
        raise ValueError(
            f"{value} is not implemented. Implemented neural networks are: "
            + ", ".join([repr(m.value) for m in cls])
        )


class NetworkConfig(BaseModel, ABC):
    """Base config class to configure neural networks."""

    # pydantic config
    model_config = ConfigDict(
        validate_assignment=True,
        use_enum_values=True,
        validate_default=True,
    )

    @computed_field
    @property
    @abstractmethod
    def network(self) -> ImplementedNetworks:
        """The name of the network."""

    @classmethod
    def base_validator_dropout(cls, v):
        """Checks that dropout is between 0 and 1."""
        if isinstance(v, float):
            assert (
                0 <= v <= 1
            ), f"dropout must be between 0 and 1 but it has been set to {v}."
        return v

    @field_validator("kernel_size", "up_kernel_size")
    @classmethod
    def base_is_odd(cls, value, field):
        """Checks if a field is odd."""
        if value != DefaultFromLibrary.YES:
            if isinstance(value, int):
                value_ = (value,)
            else:
                value_ = value
            for v in value_:
                assert v % 2 == 1, f"{field.field_name} must be odd."
        return value

    @classmethod
    def base_at_least_2d(cls, v, ctx):
        """Checks that a tuple has at least a length of two."""
        if isinstance(v, tuple):
            assert (
                len(v) >= 2
            ), f"{ctx.field_name} should have at least two dimensions (with the first one for the channel)."
        return v

    @model_validator(mode="after")
    def base_model_validator(self):
        """Checks coherence between parameters."""
        if self.kernel_size != DefaultFromLibrary.YES:
            assert self._check_dimensions(
                self.kernel_size
            ), f"You must passed an int or a sequence of {self.dim} ints (the dimensionality of your images) for kernel_size. You passed {self.kernel_size}."
        if self.up_kernel_size != DefaultFromLibrary.YES:
            assert self._check_dimensions(
                self.up_kernel_size
            ), f"You must passed an int or a sequence of {self.dim} ints (the dimensionality of your images) for up_kernel_size. You passed {self.up_kernel_size}."
        return self

    def _check_dimensions(
        self,
        value: Union[float, Tuple[float, ...]],
    ) -> bool:
        """Checks if a tuple has the right dimension."""
        if isinstance(value, tuple):
            return len(value) == self.dim
        return True


class PreTrainedConfig(NetworkConfig):
    """Base config class for SOTA networks."""

    num_outputs: Optional[PositiveInt]
    output_act: Union[
        Optional[ActivationParameters], DefaultFromLibrary
    ] = DefaultFromLibrary.YES
    pretrained: Union[bool, DefaultFromLibrary] = DefaultFromLibrary.YES
