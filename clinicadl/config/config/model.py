from logging import getLogger

from pydantic import BaseModel, ConfigDict, field_validator
from pydantic.types import NonNegativeFloat

logger = getLogger("clinicadl.model_config")


class ModelConfig(BaseModel):  # TODO : put in model module
    """
    Abstract config class for the model.

    architecture and loss are specific to the task, thus they need
    to be specified in a subclass.
    """

    architecture: str
    dropout: NonNegativeFloat = 0.0
    loss: str
    multi_network: bool = False
    selection_threshold: float = 0.0
    # pydantic config
    model_config = ConfigDict(validate_assignment=True)

    @field_validator("dropout")
    def validator_dropout(cls, v):
        assert (
            0 <= v <= 1
        ), f"dropout must be between 0 and 1 but it has been set to {v}."
        return v
