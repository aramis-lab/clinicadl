from logging import getLogger
from typing import Optional

from pydantic import computed_field

from clinicadl.data.config.file_type import FileType
from clinicadl.data.old_preprocessing import Preprocessing
from clinicadl.utils.enum import PreprocessingMethod

logger = getLogger("clinicadl.preprocessing.custom")


class RawCustom:
    pass


class PreprocessingCustom(Preprocessing):
    """
    Configuration for custom preprocessing with a user-defined suffix.
    """

    custom_suffix: str = ""

    @computed_field
    @property
    def preprocessing(self) -> PreprocessingMethod:
        """The preprocessing method."""
        return PreprocessingMethod.CUSTOM

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
        return f"Preprocessing of custom images with suffix {self.custom_suffix} "
