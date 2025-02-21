from ..file_type import FileType
from ..modalities import PET
from .base import RawData


class RawPET(RawData, PET):
    """
    Configuration class for raw Positron Emission Tomography (PET) images.

    This class represents PET images in their raw BIDS format and provides
    methods to define file patterns and descriptions.
    """

    def _get_bids_filetype(self) -> FileType:
        """
        Generate the BIDS-compatible file type pattern and description.

        Returns:
            FileType: A FileType object containing the pattern and description.
        """
        description = f"Raw PET NIfTI images with tracer '{self.tracer}'"
        trc = f"trc-{self.tracer}"
        rec = ""
        if self.reconstruction:
            description += f" and reconstruction method '{self.reconstruction}'"
            rec = f"_rec-{self.reconstruction}"

        return FileType(
            pattern=f"pet/sub-*_ses-*_{trc}{rec}_pet.nii*", description=description
        )
