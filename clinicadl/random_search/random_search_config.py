from collections.abc import Iterable
from enum import Enum
from typing import Any, Dict, List

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, field_validator

from clinicadl.train.tasks import ClassificationConfig as BaseClassificationConfig
from clinicadl.train.tasks import RegressionConfig as BaseRegressionConfig
from clinicadl.train.trainer import ModelConfig as BaseModelConfig


class Normalization(str, Enum):  # TODO : put in model module
    """Available normalization layers in ClinicaDL."""

    BATCH = "batch"
    GROUP = "group"
    INSTANCE = "instance"


class Pooling(str, Enum):  # TODO : put in model module
    """Available pooling techniques in ClinicaDL."""

    MAXPOOLING = "MaxPooling"
    STRIDE = "stride"


class ModelConfig(BaseModelConfig):  # TODO : put in model module
    """Config class for Random Search models."""

    architecture: str = Field(default="RandomArchitecture", frozen=True)
    convolutions_dict: Dict[str, Any]
    n_fcblocks: PositiveInt
    network_normalization: Normalization = Normalization.BATCH


class RandomSearchConfig(
    BaseModel
):  # TODO : add fields for all parameters that can be sampled
    """Config class to perform Random Search."""

    channels_limit: PositiveInt = 512
    d_reduction: List[Pooling] = Pooling.MAXPOOLING
    first_conv_width: List[PositiveInt]
    n_conv: PositiveInt = 1
    n_convblocks: List[PositiveInt]
    n_fcblocks: List[PositiveInt]
    network_normalization: List[Normalization] = Normalization.BATCH
    wd_bool: bool = True
    # pydantic config
    model_config = ConfigDict(validate_assignment=True)

    @field_validator(
        "d_reduction",
        "first_conv_width",
        "n_convblocks",
        "n_fcblocks" "network_normalization",
        mode="before",
    )
    def fixed_to_list(cls, v):
        """Transforms fixed parameters to lists of length 1."""
        if not isinstance(v, Iterable):
            return [v]
        return v


def model_config_random_search(base_model_config):
    class ModelConfig(base_model_config):
        """Config class for Random Search models."""

        architecture: str = Field(default="RandomArchitecture", frozen=True)
        convolutions_dict: Dict[str, Any]
        n_fcblocks: PositiveInt
        network_normalization: Normalization = Normalization.BATCH

    return ModelConfig


@model_config_random_search
class ClassificationConfig(BaseClassificationConfig):
    pass
