from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Type, Union

from pydantic import BaseModel, ConfigDict, PositiveInt, field_validator

from clinicadl.config.config.pipelines.task.classification import (
    ClassificationConfig as BaseClassificationConfig,
)
from clinicadl.config.config.pipelines.task.regression import (
    RegressionConfig as BaseRegressionConfig,
)
from clinicadl.config.config_utils import get_type_from_config_class as get_type
from clinicadl.utils.enum import Normalization, Pooling, Task

if TYPE_CHECKING:
    from clinicadl.trainer.trainer import TrainConfig


class RandomSearchConfig(
    BaseModel
):  # TODO : add fields for all parameters that can be sampled
    """
    Config class to perform Random Search.

    The user must specified at least the following arguments:
    - first_conv_width
    - n_convblocks
    - n_fcblocks
    """

    channels_limit: PositiveInt = 512
    d_reduction: Tuple[Pooling, ...] = (Pooling.MAXPOOLING,)
    first_conv_width: Tuple[PositiveInt, ...]
    n_conv: PositiveInt = 1
    n_convblocks: Tuple[PositiveInt, ...]
    n_fcblocks: Tuple[PositiveInt, ...]
    network_normalization: Tuple[Optional[Normalization], ...] = (
        Normalization.BATCH,
    )  # TODO : change name to be consistent?
    wd_bool: bool = True
    # pydantic config
    model_config = ConfigDict(validate_assignment=True)

    @field_validator(
        "d_reduction",
        "first_conv_width",
        "n_convblocks",
        "n_fcblocks",
        "network_normalization",
        mode="before",
    )
    def to_tuple(cls, v):
        """Transforms fixed parameters to tuples of length 1 and lists to tuples."""
        if not isinstance(v, (tuple, list)):
            return (v,)
        elif isinstance(v, list):
            return tuple(v)
        return v


def training_config_for_random_models(base_training_config):
    base_model_config = get_type("model", base_training_config)

    class NetworkConfig(base_model_config):
        """Config class for random models."""

        architecture: str = "RandomArchitecture"
        convolutions_dict: Dict[str, Any]  # TODO : be more precise?
        n_fcblocks: PositiveInt
        network_normalization: Optional[Normalization] = Normalization.BATCH

        @field_validator("architecture")
        def architecture_validator(cls, v):
            assert (
                v == "RandomArchitecture"
            ), "Only RandomArchitecture can be used in Random Search."

    class TrainConfig(base_training_config):
        """
        Config class for the training of a random model.

        The user must specified at least the following arguments:
            - caps_directory
            - preprocessing_json
            - tsv_directory
            - output_maps_directory
            - convolutions_dict
            - n_fcblocks
        """

        model: NetworkConfig

    return TrainConfig


@training_config_for_random_models
class ClassificationConfig(BaseClassificationConfig):
    pass


@training_config_for_random_models
class RegressionConfig(BaseRegressionConfig):
    pass


def create_training_config(task: Union[str, Task]) -> Type[TrainConfig]:
    """
    A factory function to create a Training Config class suited for the task, in Random Search mode.

    Parameters
    ----------
    task : Union[str, Task]
        The Deep Learning task (e.g. classification).

    Returns
    -------
    Type[TrainConfig]
        The Config class.
    """
    task = Task(task)
    if task == Task.CLASSIFICATION:
        return ClassificationConfig
    elif task == Task.REGRESSION:
        return RegressionConfig
    elif task == Task.RECONSTRUCTION:
        raise ValueError("Random Search not yet implemented for Reconstruction.")
