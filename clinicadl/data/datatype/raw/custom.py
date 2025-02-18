from ..file_type import FileType
from ..modalities import Custom as CustomModality
from .base import RawData


class RawCustom(RawData, CustomModality):
    """
    Configuration class to handle raw custom imaging data with a user-defined suffix.
    """

    def _get_bids_filetype(self) -> FileType:
        """
        Generate the BIDS-compatible file type pattern and description.

        Returns:
            FileType: A FileType object containing the pattern and description.
        """
        return FileType(
            pattern=f"sub-*_ses-*_{self.custom_suffix}.nii*",
            description=f"Raw custom NIfTI images with suffix '{self.custom_suffix}'",
        )
