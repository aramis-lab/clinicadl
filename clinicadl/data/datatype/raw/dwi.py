from typing import Optional

from clinicadl.data.datatype.file_type import FileType
from clinicadl.data.datatype.modalities import DWI

from .base import RawData


class RawDWI(RawData, DWI):
    """
    Configuration class for raw Diffusion Weighted Imaging (DWI) data.

    This class represents DWI images in their raw BIDS format and provides
    methods to define file patterns and descriptions.
    """

    def _get_bids_filetype(self) -> FileType:
        """
        Get the BIDS-compatible file type for raw DWI images.

        Args:
            reconstruction (Optional[str]): Reconstruction identifier (unused in this implementation).

        Returns:
            FileType: A FileType object with the file pattern and description.
        """
        return FileType(
            pattern="dwi/sub-*_ses-*_dwi.nii*",
            description="DWI NIfTI",
        )

    def __str__(self) -> str:
        """
        String representation of the RawDWI class.
        """
        return "Configuration for raw DWI images in BIDS format."
