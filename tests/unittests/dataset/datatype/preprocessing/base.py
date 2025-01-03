import abc
from enum import Enum

from pydantic import computed_field

from clinicadl.data.datatype.file_type import FileType
from clinicadl.data.datatype.modalities import ImageModality
from clinicadl.utils.config import ClinicaDLConfig


class PreprocessingMethod(str, Enum):
    """Possible preprocessing methods available in Clinica."""

    T1_LINEAR = "t1-linear"
    PET_LINEAR = "pet-linear"
    FLAIR_LINEAR = "flair-linear"
    CUSTOM = "custom"
    DWI_DTI = "dwi-dti"


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
        return self.get_caps_filetype()

    @abc.abstractmethod
    def get_caps_filetype(self) -> FileType:
        """
        Abstract method to obtain FileType details.

        The specific implementation of this method should return a FileType
        object based on the preprocessing pipeline and modality.
        """
        pass


class _PreprocessingWithCrop(Preprocessing):
    """
    Base class for preprocessing methods with the option to use uncropped images.

    If the `use_uncropped_image` is set to True, it uses the uncropped image pattern;
    otherwise, it adds the '_desc-Crop' suffix to the pattern to indicate cropped images.
    """

    use_uncropped_image: bool = False

    def linear_nii(
        self, modality: ImageModality, needed_pipeline: PreprocessingMethod
    ) -> FileType:
        """
        Constructs the file type for linear preprocessed image data in CAPS format.

        Args:
            modality (ImageModality): The image modality (e.g., T1w, DWI, etc.)
            needed_pipeline (PreprocessingMethod): The preprocessing pipeline applied to the modality.

        Returns:
            FileType: A `FileType` object that describes the preprocessed file type.
        """
        # Determine the suffix based on the uncropped image option
        desc_crop = "" if self.use_uncropped_image else "_desc-Crop"

        # Construct the pattern based on the preprocessing method and modality
        pattern = f"{self.preprocessing.value.replace('-', '_')}/*space-MNI152NLin2009cSym{desc_crop}_res-1x1x1_{modality.value}.nii.gz"

        # Construct the description based on the uncropped image option
        description = f"{modality.value} Image registered in MNI152NLin2009cSym space using {needed_pipeline.value} pipeline"
        if not self.use_uncropped_image:
            description += (
                " and cropped (matrix size 169×208×179, 1 mm isotropic voxels)"
            )

        # Return the constructed FileType
        return FileType(
            pattern=pattern,
            description=description,
            needed_pipeline=needed_pipeline,
        )
