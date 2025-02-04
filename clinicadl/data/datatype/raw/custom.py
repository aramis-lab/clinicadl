from ..modalities import Custom
from ..preprocessing.file_type import FileType
from .base import RawData


class RawCustom(RawData, Custom):
    """
    Configuration class for raw custom imaging data with a user-defined suffix.

    This class represents custom images in their raw BIDS format and provides
    methods to define file patterns and descriptions.
    """

    def _get_bids_filetype(self) -> FileType:
        """
        Generate the BIDS-compatible file type pattern and description.

        Args:
            reconstruction (Optional[str]): Reconstruction identifier (unused here).

        Returns:
            FileType: A FileType object containing the pattern and description.
        """
        return FileType(
            pattern=f"*{self.custom_suffix}",
            description="Custom suffix for raw data",
        )

    def __str__(self) -> str:
        """
        String representation of the RawCustom class.
        """
        return f"Custom raw images with suffix '{self.custom_suffix}'"
