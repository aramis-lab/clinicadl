from pydantic import computed_field

from clinicadl.data.datatype.file_type import FileType
from clinicadl.data.datatype.modalities import Custom as CustomModality

from .base import Preprocessing, PreprocessingMethod


class Custom(Preprocessing, CustomModality):
    """
    Configuration for custom preprocessing with a user-defined suffix.
    """

    @computed_field
    @property
    def preprocessing(self) -> PreprocessingMethod:
        """The preprocessing method."""
        return PreprocessingMethod.CUSTOM

    def get_caps_filetype(self) -> FileType:
        """
        Constructs the FileType for custom preprocessing.
        """
        return FileType(
            pattern=f"custom/*{self.custom_suffix}",
            description="Custom suffix",
        )

    def __str__(self):
        """
        Provides a string representation of the preprocessing configuration.
        """
        return f"Preprocessing of custom images with suffix {self.custom_suffix} "
