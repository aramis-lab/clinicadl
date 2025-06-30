import abc

from pydantic import computed_field

from clinicadl.dictionary.suffixes import JSON, TSV
from clinicadl.utils.config import ClinicaDLConfig

from ..file_type import FileType
from ..modalities import Modality


class Preprocessing(ClinicaDLConfig, abc.ABC):
    """
    Abstract configuration class to model the preprocessing step.

    This class should be inherited by all preprocessing methods to define specific
    configurations for each preprocessing pipeline.
    """

    @computed_field
    @property
    @abc.abstractmethod
    def name(self) -> str:
        """The preprocessing method being applied (e.g., t1-linear, pet-linear)."""

    @computed_field
    @property
    def file_type(self) -> FileType:
        """
        Returns the FileType associated with the preprocessed data.
        This method delegates to the `_get_caps_filetype()`.
        """
        return self._get_caps_filetype()

    @property
    def tsv_filename(self) -> str:
        """
        Builds a filename for a tsv file saving
        information on this preprocessing.
        """
        return "overview_" + self._get_file_name() + TSV

    @property
    def json_filename(self) -> str:
        """
        Builds a filename for a json file saving
        information on this preprocessing.
        """
        return "default_" + self._get_file_name() + JSON

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

    @abc.abstractmethod
    def _get_file_name(self) -> str:
        """
        Builds a suffix for files saving
        information on this preprocessing.
        """


class _LinearPreprocessing(Preprocessing, Modality):
    """
    Base class for linear preprocessings (`t1-linear`, `flair-linear` or `pet-linear`).

    If the `use_uncropped_image` is set to True, it uses the uncropped image pattern;
    otherwise, it adds the `_desc-Crop` suffix to the pattern to select cropped images.
    """

    use_uncropped_image: bool = False

    def _get_file_pattern(self) -> str:
        """
        Constructs the file pattern depending on the preprocessing parameters.
        May be overwritten for some preprocessings.
        """
        desc_crop = "" if self.use_uncropped_image else "_desc-Crop"
        return f"sub-*_ses-*_space-MNI152NLin2009cSym{desc_crop}_res-1x1x1_{self.modality}.nii*"

    def _get_description(self) -> str:
        """
        Constructs a description depending on the preprocessing parameters.
        May be overwritten for some preprocessings.
        """
        modality = self.modality
        if not modality.endswith("w"):
            modality = modality.upper()

        description = f"{modality} images registered to MNI152NLin2009cSym space using Clinica's '{self.name}' pipeline"

        if not self.use_uncropped_image:
            description += (
                ", and cropped (matrix size 169×208×179, 1 mm isotropic voxels)"
            )
        return description

    def _get_caps_filetype(self) -> FileType:
        """
        Base method to construct the FileType for linear preprocessings.
        """
        pattern = self._get_file_pattern()
        pattern = self.name.replace("-", "_") + f"/{pattern}"
        description = self._get_description()

        return FileType(
            pattern=pattern,
            description=description,
            needed_pipeline=self.name,
        )

    def _get_file_name(self) -> str:
        """
        Builds a suffix for files saving
        information on this preprocessing.
        """
        return f"{self.name}{'' if self.use_uncropped_image else '_cropped'}"
