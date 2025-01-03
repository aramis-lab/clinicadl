from pydantic import computed_field, field_validator

from .base import ImageModality, Modality


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
        if not value:
            raise ValueError("Custom suffix cannot be empty.")
        return str(value)


# Create an instance of RawCustom
custom_data = Custom(custom_suffix="example")

# Access its properties
print(custom_data.modality)  # Output: ImageModality.CUSTOM
print(custom_data.custom_suffix)  # Output: "example"

# String representation
print(
    custom_data
)  # Output: RawCustom Configuration: Custom raw images with suffix 'example'.

# Invalid suffix (triggers validation error)
try:
    invalid_custom = Custom(custom_suffix="!invalid")
except ValueError as e:
    print(e)  # Output: Custom suffix must be alphanumeric. Avoid special characters.
