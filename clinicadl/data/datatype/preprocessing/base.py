import abc

from pydantic import computed_field

from clinicadl.data.datatype.file_type import FileType
from clinicadl.data.datatype.modalities import Modality
from clinicadl.data.datatype.utils import PreprocessingMethod
from clinicadl.utils.config import ClinicaDLConfig


class Preprocessing(ClinicaDLConfig, abc.ABC):
    """
    Abstract configuration class for the preprocessing procedure.

    This class should be inherited by all preprocessing methods to define specific
    configurations for each preprocessing pipeline.
    """

    @computed_field
    @property
    @abc.abstractmethod
    def preprocessing(self) -> PreprocessingMethod:
        """The preprocessing method being applied (e.g., t1-linear, pet-linear)."""

    @computed_field
    @property
    def file_type(self) -> FileType:
        """
        Returns the FileType associated with the preprocessed data.
        This method delegates to the `get_caps_filetype()` method to get the details.
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

    def _get_caps_filetype(self) -> FileType:
        """
        Base method to construct the FileType for linear preprocessings.
        """
        # Determine the suffix based on the uncropped image option
        desc_crop = "" if self.use_uncropped_image else "_desc-Crop"

        # Construct the pattern based on the preprocessing method and modality
        pattern = f"{self.preprocessing.value.replace('-', '_')}/*space-MNI152NLin2009cSym{desc_crop}_res-1x1x1_{self.modality.value}.nii.gz"

        # Construct the description based on the uncropped image option
        description = f"{self.modality.value} Image registered in MNI152NLin2009cSym space using {self.preprocessing.value} pipeline"
        if not self.use_uncropped_image:
            description += (
                " and cropped (matrix size 169×208×179, 1 mm isotropic voxels)"
            )

        # Return the constructed FileType
        return FileType(
            pattern=pattern,
            description=description,
            needed_pipeline=self.preprocessing.value,
        )
