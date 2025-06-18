from ..file_type import FileType
from ..modalities import Flair
from .base import RawData


class RawFlair(RawData, Flair):
    """
    Configuration class to handle raw FLAIR (Fluid-Attenuated Inversion Recovery) T2-weighted MRI images.
    """

    def _get_bids_filetype(self) -> FileType:
        """
        Generate the BIDS-compatible file type pattern and description.

        Returns:
            FileType: A FileType object containing the pattern and description.
        """
        return FileType(
            pattern="anat/sub-*_ses-*_FLAIR.nii*",
            description="Raw FLAIR T2w MRI NIfTI images",
        )
