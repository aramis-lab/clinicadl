import abc

from pydantic import computed_field

from clinicadl.utils.config import ClinicaDLConfig

from ..file_type import FileType
from ..modalities import Modality


class Preprocessing(ClinicaDLConfig, abc.ABC):
    """
    Abstract configuration class for to model the preprocessing step.

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

    def __str__(self):
        """
        Provides a string representation of the preprocessing.
        """
        return self.file_type.description

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
    otherwise, it adds the `_desc-Crop` suffix to the pattern to select cropped images.
    """

    use_uncropped_image: bool = False

    def _get_filename(self) -> str:
        """
        Constructs the file name depending on the preprocessing parameters.
        May be overwritten for some preprocessings.
        """
        desc_crop = "" if self.use_uncropped_image else "_desc-Crop"
        return f"sub-*_ses-*_space-MNI152NLin2009cSym{desc_crop}_res-1x1x1_{self.modality.value}.nii*"

    def _get_description(self) -> str:
        """
        Constructs a description depending on the preprocessing parameters.
        May be overwritten for some preprocessings.
        """
        modality = self.modality.value
        if not modality.endswith("w"):
            modality = modality.upper()

        description = f"{modality} images registered to MNI152NLin2009cSym space using Clinica's '{self.preprocessing}' pipeline"

        if not self.use_uncropped_image:
            description += (
                ", and cropped (matrix size 169×208×179, 1 mm isotropic voxels)"
            )
        return description

    def _get_caps_filetype(self) -> FileType:
        """
        Base method to construct the FileType for linear preprocessings.
        """
        filename = self._get_filename()
        pattern = self.preprocessing.replace("-", "_") + f"/{filename}"
        description = self._get_description()

        return FileType(
            pattern=pattern,
            description=description,
            needed_pipeline=self.preprocessing,
        )
