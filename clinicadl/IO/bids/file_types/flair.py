from pydantic import computed_field

from ...enum import ImageModality
from ...file_type import FileType


class FlairFileType(FileType):
    """
    Configuration class to handle raw custom imaging data with a user-defined suffix.
    """

    def __init__(self):
        """
        Generate the BIDS-compatible file type pattern and description.

        Returns:
            FileType: A FileType object containing the pattern and description.
        """

        description = "Raw FLAIR T2w MRI NIfTI images"
        super().__init__(container="anat", description=description)

    @computed_field
    @property
    def modality(self) -> str:
        """
        The modality, always 'custom' here.
        """
        return ImageModality.FLAIR.value
