import abc
from logging import getLogger
from typing import Optional

from pydantic import computed_field

from clinicadl.utils.config import ClinicaDLConfig
from clinicadl.utils.enum import LinearModality, PreprocessingMethod
from clinicadl.utils.iotools.clinica_utils import FileType

logger = getLogger("clinicadl.preprocessing.base")


class Preprocessing(ClinicaDLConfig, abc.ABC):
    """
    Abstract config class for the preprocessing procedure.
    """

    @computed_field
    @property
    @abc.abstractmethod
    def preprocessing(self) -> PreprocessingMethod:
        """The preprocessing method."""

    @computed_field
    @property
    def file_type(self) -> FileType:
        return self.get_caps_filetype()

    @abc.abstractmethod
    def get_bids_filetype(self, reconstruction: Optional[str] = None) -> FileType:
        """Abstract method to get the BIDS filetype."""

    @abc.abstractmethod
    def get_caps_filetype(self) -> FileType:
        """Abstract method to obtain FileType details."""


class _PreprocessingWithCrop(Preprocessing):
    """
    Base class for the preprocessings with the option 'use_uncropped_image.
    """

    use_uncropped_image: bool = True

    def linear_nii(
        self, modality: LinearModality, needed_pipeline: PreprocessingMethod
    ) -> FileType:
        """
        Constructs the file type for linear caps image data
        """
        desc_crop = "" if self.use_uncropped_image else "_desc-Crop"

        file_type = FileType(
            pattern=f"{self.preprocessing.value.replace('-', '_')}/*space-MNI152NLin2009cSym{desc_crop}_res-1x1x1_{modality.value}.nii.gz",
            description=f"{modality.value} Image registered in MNI152NLin2009cSym space using {needed_pipeline.value} pipeline "
            + (
                ""
                if self.use_uncropped_image
                else "and cropped (matrix size 169×208×179, 1 mm isotropic voxels)"
            ),
            needed_pipeline=needed_pipeline,
        )
        return file_type
