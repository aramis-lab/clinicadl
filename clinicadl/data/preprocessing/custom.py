from logging import getLogger
from pathlib import Path
from typing import Optional

from clinicadl.data.preprocessing.base import BasePreprocessing
from clinicadl.utils.enum import Preprocessing
from clinicadl.utils.iotools.clinica_utils import FileType

logger = getLogger("clinicadl.preprocessing.custom")


class PreprocessingCustom(BasePreprocessing):
    """
    Configuration for custom preprocessing with a user-defined suffix.
    """

    custom_suffix: str = ""
    preprocessing: Preprocessing = Preprocessing.CUSTOM

    def get_bids_filetype(self, reconstruction: Optional[str] = None) -> FileType:
        return FileType(
            pattern=f"*{self.custom_suffix}",
            description="Custom suffix",
        )

    def get_caps_filetype(self) -> FileType:
        return FileType(
            pattern=f"custom/*{self.custom_suffix}",
            description="Custom suffix",
        )

    def __str__(self):
        return f"Preprocessing of {'uncropped' if self.use_uncropped_image else 'cropped'} custom images with suffix {self.custom_suffix} "
