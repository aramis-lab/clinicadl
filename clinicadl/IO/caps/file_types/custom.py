from ...file_type import FileType
from ...modalities import Custom


class CustomFileType(FileType, Custom):
    """
    Configuration class to handle raw custom imaging data with a user-defined suffix.
    """

    def __init__(self, custom_suffix: str):
        """
        Generate the BIDS-compatible file type pattern and description.

        Returns:
            FileType: A FileType object containing the pattern and description.
        """
        self.custom_suffix = custom_suffix
        super().__init__()

    @property
    def description(self) -> str:
        """
        The description of the file type.
        """
        return f"Raw custom NIfTI images with suffix '{self.custom_suffix}'"

    @property
    def container(self) -> str:
        """
        The name of the folder where the file is stored.
        """
        return self.modality
