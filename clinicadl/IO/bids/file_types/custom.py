from pydantic import computed_field

from ...enum import ImageModality
from ...file_type import FileType


class CustomFileType(FileType):
    """
    Configuration class to handle raw custom imaging data with a user-defined suffix.
    """

    def __init__(self, custom_suffix: str):
        """
        Generate the BIDS-compatible file type pattern and description.

        Returns:
            FileType: A FileType object containing the pattern and description.
        """
        description = f"Raw custom NIfTI images with suffix '{custom_suffix}'"
        super().__init__(
            container=self.modality, pattern=custom_suffix, description=description
        )

    @computed_field
    @property
    def modality(self) -> str:
        """
        The modality, always 'custom' here.
        """
        return ImageModality.CUSTOM.value
