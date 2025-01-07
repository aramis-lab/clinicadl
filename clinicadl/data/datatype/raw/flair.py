from typing import Optional

from clinicadl.data.datatype.file_type import FileType
from clinicadl.data.datatype.modalities import Flair

from .base import RawData


class RawFlair(RawData, Flair):
    """
    Configuration class for raw FLAIR (Fluid-Attenuated Inversion Recovery) T2-weighted MRI images.

    This class represents flair images in their raw BIDS format and provides
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
        return FileType(pattern="sub-*_ses-*_flair.nii*", description="FLAIR T2w MRI")

    def __str__(self):
        """
        String representation of the RawFlair class.
        """
        return "Raw flair T2w MRI images."
