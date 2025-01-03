from typing import Optional

from clinicadl.data.datatype.file_type import FileType
from clinicadl.data.datatype.modalities import T1w

from .base import RawData


class RawT1w(RawData, T1w):
    """
    Configuration class for raw T1-weighted MRI (T1w) images.

    This class represents T1w images in their raw BIDS format and provides
    methods to define file patterns and descriptions.
    """

    def get_bids_filetype(self, reconstruction: Optional[str] = None) -> FileType:
        """
        Generate the BIDS-compatible file type pattern and description.

        Args:
            reconstruction (Optional[str]): Reconstruction identifier (unused here).

        Returns:
            FileType: A FileType object containing the pattern and description.
        """
        return FileType(pattern="anat/sub-*_ses-*_T1w.nii*", description="T1w MRI")

    def __str__(self):
        """
        String representation of the RawT1w class.
        """
        return "Raw T1w MRI images."
