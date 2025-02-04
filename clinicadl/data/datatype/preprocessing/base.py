import abc

from pydantic import computed_field

from clinicadl.utils.config import ClinicaDLConfig

from ..modalities import Modality
from .file_type import FileType


class Preprocessing(ClinicaDLConfig, abc.ABC):
    """
    Abstract configuration class for the preprocessing procedure.

    This class should be inherited by all preprocessing methods to define specific
    configurations for each preprocessing pipeline.
    """

    @computed_field
    @property
    @abc.abstractmethod
    def preprocessing(self) -> str:
        """The preprocessing method being applied (e.g., t1-linear, pet-linear)."""

    @computed_field
    @property
    def file_type(self) -> FileType:
        """
        Returns the FileType associated with the preprocessed data.
        This method delegates to the `_get_caps_filetype()`.
        """
        return self._get_caps_filetype()

    @abc.abstractmethod
    def _get_caps_filetype(self) -> FileType:
        """
        Abstract method to obtain FileType details.

        The specific implementation of this method should return a FileType
        object based on the preprocessing pipeline and modality.
        """


class _LinearPreprocessing(Preprocessing, Modality):
    """
    Base class for linear preprocessings (`t1-linear`, `flair-linear` or `pet-linear`).

    If the `use_uncropped_image` is set to True, it uses the uncropped image pattern;
    otherwise, it adds the `_desc-Crop` suffix to the pattern to indicate cropped images.
    """

    use_uncropped_image: bool = False

    def _get_filename(self) -> str:
        """
        Constructs the file name depending on the preprocessing parameters.
        May be overwritten for some preprocessings.
        """
        desc_crop = "" if self.use_uncropped_image else "_desc-Crop"
        return (
            f"*space-MNI152NLin2009cSym{desc_crop}_res-1x1x1_{self.modality.value}.nii*"
        )

    def _get_caps_filetype(self) -> FileType:
        """
        Base method to construct the FileType for linear preprocessings.
        """
        # Construct the pattern based on the preprocessing method and modality
        filename = self._get_filename()
        pattern = self.preprocessing.replace("-", "_") + f"/{filename}"

        # Construct the description based on the uncropped image option
        description = f"{self.modality.value} image registered in MNI152NLin2009cSym space using {self.preprocessing} pipeline"

        if not self.use_uncropped_image:
            description += (
                " and cropped (matrix size 169×208×179, 1 mm isotropic voxels)"
            )

        # Return the constructed FileType
        return FileType(
            pattern=pattern,
            description=description,
            needed_pipeline=self.preprocessing,
        )
