import abc
from logging import getLogger
from typing import Optional

from pydantic import BaseModel, ConfigDict, computed_field

from clinicadl.utils.enum import LinearModality, Preprocessing
from clinicadl.utils.iotools.clinica_utils import FileType

logger = getLogger("clinicadl.preprocessing.base")


class BasePreprocessing(BaseModel, abc.ABC):
    """
    Abstract config class for the preprocessing procedure.
    """

    preprocessing: Preprocessing
    use_uncropped_image: bool = False

    model_config = ConfigDict(validate_assignment=True, arbitrary_types_allowed=True)

    def get_filetype(self, bids: bool = False) -> FileType:
        return self.get_bids_filetype() if bids else self.get_caps_filetype()

    @abc.abstractmethod
    def get_bids_filetype(self, reconstruction: Optional[str] = None) -> FileType:
        """Abstract method to get the BIDS filetype."""
        pass

    @abc.abstractmethod
    def get_caps_filetype(self) -> FileType:
        """Abstract method to obtain FileType details."""
        pass

    @computed_field
    @property
    def file_type(self) -> FileType:
        if self.preprocessing not in Preprocessing:
            raise NotImplementedError(
                f"Extraction of preprocessing {self.preprocessing.value} is not implemented from CAPS directory."
            )
        else:
            return self.get_filetype()

    def linear_nii(
        self, modality: LinearModality, needed_pipeline: Preprocessing
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
