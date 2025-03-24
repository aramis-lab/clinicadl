from ..file_type import FileType
from ..modalities import DWI
from .base import RawData


class RawDWI(RawData, DWI):
    """
    Configuration class to handle raw Diffusion Weighted Imaging (DWI) data.
    """

    def _get_bids_filetype(self) -> FileType:
        """
        Get the BIDS-compatible file type for raw DWI images.

        Returns:
            FileType: A FileType object with the file pattern and description.
        """
        return FileType(
            pattern="dwi/sub-*_ses-*_dwi.nii*",
            description="Raw DW MRI NIfTI images",
        )
