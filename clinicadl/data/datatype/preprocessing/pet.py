from pydantic import computed_field

from clinicadl.data.datatype.file_type import FileType
from clinicadl.data.datatype.modalities import PET

from .base import PreprocessingMethod, _PreprocessingWithCrop


class PETLinear(_PreprocessingWithCrop, PET):
    """Config class for Clinica's 'pet-linear' preprocessing."""

    @computed_field
    @property
    def preprocessing(self) -> PreprocessingMethod:
        """The preprocessing method."""
        return PreprocessingMethod.PET_LINEAR

    def get_caps_filetype(self) -> FileType:
        """
        Constructs the FileType for pet-linear preprocessing.
        """

        des_crop = "" if self.use_uncropped_image else "_desc-Crop"

        description = f"{self.modality.value} Image registered in MNI152NLin2009cSym space using pet-linear pipeline"

        if not self.use_uncropped_image:
            description += (
                " and cropped (matrix size 169×208×179, 1 mm isotropic voxels)"
            )
        return FileType(
            pattern=f"pet_linear/*_trc-{self.tracer}_space-MNI152NLin2009cSym{des_crop}_res-1x1x1_suvr-{self.suvr_reference_region}_pet.nii.gz",
            description=description,
            needed_pipeline=PreprocessingMethod.PET_LINEAR,
        )

    def __str__(self):
        """
        Provides a string representation of the preprocessing configuration.
        """
        return f"Preprocessing of {'uncropped' if self.use_uncropped_image else 'cropped'} PET images with tracer {self.tracer} and suvr reference region {self.suvr_reference_region}. "
