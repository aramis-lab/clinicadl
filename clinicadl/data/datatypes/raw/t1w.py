from ..file_type import FileType
from ..modalities import T1w
from .base import RawData


class RawT1w(RawData, T1w):
    """
    Configuration class for raw T1-weighted MRI (T1w) images.

    This class represents T1w images in their raw BIDS format and provides
    methods to define file patterns and descriptions.
    """

    def _get_bids_filetype(self) -> FileType:
        """
        Generate the BIDS-compatible file type pattern and description.

        Returns:
            FileType: A FileType object containing the pattern and description.
        """
        return FileType(
            pattern="anat/sub-*_ses-*_T1w.nii*", description="Raw T1w MRI NIfTI images"
        )
