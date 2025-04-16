from typing import Optional

from pydantic import computed_field

from ...enum import ImageModality
from ...file_type import FileType


class PETFileType(FileType):
    """
    Configuration class to handle raw custom imaging data with a user-defined suffix.
    """

    def __init__(self, tracer: str, reconstruction: Optional[str] = None):
        """
        Generate the BIDS-compatible file type pattern and description.

        Returns:
            FileType: A FileType object containing the pattern and description.
        """
        self.tracer = tracer
        self.reconstruction = reconstruction

        description = f"Raw PET NIfTI images with tracer '{self.tracer}'"
        trc = f"_trc-{self.tracer}"
        rec = ""
        if self.reconstruction:
            description += f" and reconstruction method '{self.reconstruction}'"
            rec = f"_rec-{self.reconstruction}"

        super().__init__(
            container=self.modality, pattern=trc + rec, description=description
        )

    @computed_field
    @property
    def modality(self) -> str:
        """
        The modality, always 'custom' here.
        """
        return ImageModality.PET.value
