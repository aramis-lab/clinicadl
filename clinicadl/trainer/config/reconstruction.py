from logging import getLogger
from typing import Tuple, Union

from pydantic import PositiveFloat, PositiveInt, computed_field, field_validator

from clinicadl.network.config import NetworkConfig as BaseNetworkConfig
from clinicadl.trainer.config.train import TrainConfig
from clinicadl.utils.enum import (
    Normalization,
    ReconstructionLoss,
    ReconstructionMetric,
    Task,
)
from clinicadl.validation.validation import ValidationConfig as BaseValidationConfig

logger = getLogger("clinicadl.reconstruction_config")


class NetworkConfig(BaseNetworkConfig):  # TODO : put in model module
    """Config class for reconstruction models."""

    architecture: str = "AE_Conv5_FC3"
    loss: ReconstructionLoss = ReconstructionLoss.MSELoss
    latent_space_size: PositiveInt = 128
    feature_size: PositiveInt = 1024
    n_conv: PositiveInt = 4
    io_layer_channels: PositiveInt = 8
    recons_weight: PositiveFloat = 1.0
    kl_weight: PositiveFloat = 1.0
    normalization: Normalization = Normalization.BATCH

    @field_validator("architecture")
    def validator_architecture(cls, v):
        return v  # TODO : connect to network module to have list of available architectures


class ValidationConfig(BaseValidationConfig):
    """Config class for the validation procedure in reconstruction mode."""

    # selection_metrics: Tuple[ReconstructionMetric, ...] = (ReconstructionMetric.LOSS,)
    selection_metrics: Tuple[Union[ReconstructionMetric, str], ...] = ("loss",)

    @field_validator("selection_metrics", mode="before")
    def list_to_tuples(cls, v):
        if isinstance(v, list):
            return tuple(v)
        return v


class ReconstructionConfig(TrainConfig):
    """
    Config class for the training of a reconstruction model.

    The user must specified at least the following arguments:
    - caps_directory
    - preprocessing_json
    - tsv_directory
    - output_maps_directory
    """

    model: NetworkConfig
    validation: ValidationConfig

    @computed_field
    @property
    def network_task(self) -> Task:
        return Task.RECONSTRUCTION
