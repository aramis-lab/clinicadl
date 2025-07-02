from logging import getLogger

from pydantic import computed_field

from ..enum import PreprocessingMethod
from ..file_type import FileType
from ..modalities import Custom as CustomModality
from .base import Preprocessing

logger = getLogger("clinicadl.data.datatypes.preprocessing.custom")


class Custom(Preprocessing, CustomModality):
    """
    Configuration class to handle custom images,
    i.e. images that have not been preprocessed with any of the supported
    ``Clinica`` pipelines (``t1-linear``, ``flair-linear``, ``pet-linear``
    and ``dwi-dti``).

    Parameters
    ----------
    custom_suffix : str
        The suffix to identify the files to select.\n
        Only the files that match the pattern ``custom/sub-*_ses-*_{custom_suffix}.nii*``
        in the :term:`CAPS` structure will be considered.
    """

    @computed_field
    @property
    def name(self) -> str:
        """The preprocessing method."""
        return PreprocessingMethod.CUSTOM.value

    def _get_caps_filetype(self) -> FileType:
        """
        Constructs the FileType for custom preprocessing.
        """
        return FileType(
            pattern=f"{self.custom_suffix}/sub-*_ses-*_{self.custom_suffix}.nii*",
            description=f"Custom images with suffix '{self.custom_suffix}'",
        )

    def _get_file_name(self) -> str:
        """
        Builds a suffix for files saving
        information on this preprocessing.
        """
        return self.custom_suffix
