from pydantic import computed_field, field_validator

from clinicadl.data.datatype.utils import ImageModality

from .base import Modality


class Custom(Modality):
    """
    Configuration for custom preprocessing with a user-defined suffix.

    Attributes:
        custom_suffix (str): User-defined suffix for custom preprocessing patterns.
    """

    custom_suffix: str = ""

    @computed_field
    @property
    def modality(self) -> ImageModality:
        """
        Specifies the modality for custom preprocessing.

        Returns:
            ImageModality: The modality, always set to ImageModality.CUSTOM.
        """
        return ImageModality.CUSTOM

    @field_validator("custom_suffix", mode="before")
    def validate_suffix(cls, value: str) -> str:
        """
        Validate the custom suffix to ensure it is not empty.

        Args:
            value (str): The custom suffix to validate.

        Returns:
            str: The validated suffix.

        Raises:
            ValueError: If the suffix is empty.
        """
        return str(value)
