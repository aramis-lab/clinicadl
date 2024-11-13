from pydantic import BaseModel, ConfigDict


class ClinicaDLConfig(BaseModel):
    """Base configuration class."""

    model_config = ConfigDict(
        validate_assignment=True, use_enum_values=True, validate_default=True
    )
