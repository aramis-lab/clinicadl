from typing import Optional

from clinicadl.data.datatype.file_type import FileType
from clinicadl.data.datatype.modalities import PET

from .base import RawData


class RawPET(RawData, PET):
    """
    Configuration class for raw Positron Emission Tomography (PET) images.

    This class represents PET images in their raw BIDS format and provides
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
        trc, rec, description = "", "", "PET data"
        if self.tracer:
            description += f" with {self.tracer} tracer"
            trc = f"_trc-{self.tracer}"
        if reconstruction:
            description += f" and reconstruction method {reconstruction}"
            rec = f"_rec-{reconstruction}"

        return FileType(pattern=f"pet/*{trc}{rec}_pet.nii*", description=description)

    def __str__(self):
        """
        String representation of the RawPET class.
        """
        return f"Raw PET images with tracer {self.tracer} and suvr reference region {self.suvr_reference_region}. "
