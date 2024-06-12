from abc import ABC, abstractmethod

from pydantic import (
    BaseModel,
    ConfigDict,
    NonNegativeFloat,
    field_validator,
    model_validator,
)


class ModelConfig(ABC, BaseModel):
    """
    Abstract base config class for ClinicaDL Models.

    network and loss are specific to the type of models
    (e.g. CNN or AE) and must be specified in subclasses.
    """

    network: str
    dropout: NonNegativeFloat = 0.0
    loss: str
    # pydantic config
    model_config = ConfigDict(
        validate_assignment=True, validate_return=True, validate_default=True
    )

    @field_validator("dropout")
    @classmethod
    def validator_dropout(cls, v):
        assert (
            0 <= v <= 1
        ), f"dropout must be between 0 and 1 but it has been set to {v}."
        return v
