from enum import Enum
from logging import getLogger

from pydantic import computed_field

from ..enum import PreprocessingMethod
from ..modalities import PET
from .base import _LinearPreprocessing

logger = getLogger("clinicadl.data.datatypes.preprocessing.pet")


class SUVRReferenceRegion(str, Enum):
    """Supported SUVR reference region in Clinica."""

    PONS = "pons"
    CEREBELLUM_PONS = "cerebellumPons"
    PONS2 = "pons2"
    CEREBELLUM_PONS2 = "cerebellumPons2"


class PETLinear(PET, _LinearPreprocessing):
    """
    Configuration class to handle Positron Emission Tomography (PET) images,
    preprocessed with `Clinica pet-linear <https://aramislab.paris.inria.fr/clinica/docs/public/latest/Pipelines/PET_Linear/>`_
    pipeline.

    Parameters
    ----------
    tracer : Tracer, default="18FFDG"
        The radioactive tracer used for acquisition, among ``11CPIB``, ``18FAV1451``, ``18FAV45``, ``18FFBB``,
        ``18FFDG`` and ``18FFMM``.
    reconstruction : Optional[ReconstructionMethod], default=None
        The method used to reconstruct the image, among ``nacstat``, ``nacdyn``, ``acstat``, ``acdyn``, ``coregdyn``,
        ``coregavg``, ``coregstd`` and ``coregiso``. Leave to ``None`` if not specified.
    suvr_reference_region : SUVRReferenceRegion, default="pons"
        The reference region used to compute SUVR, among ``pons``, ``cerebellumPons``, ``pons2`` and ``cerebellumPons2``.
    use_uncropped_image : bool, default=False
        Whether to use the uncropped images returned by ``Clinica``:\n
        - if ``use_uncropped_image=True``: only the files that match the pattern
          ``pet_linear/sub-*_ses-*_trc-{tracer}_space-MNI152NLin2009cSym_res-1x1x1_suvr-{suvr_reference_region}_pet.nii*``
          in the :term:`CAPS` structure will be considered.
        - else: only the files that match the pattern
          ``pet_linear/sub-*_ses-*_trc-{tracer}_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_suvr-{suvr_reference_region}_pet.nii*``
          in the :term:`CAPS` structure will be considered.

        .. note::
            If ``reconstruction`` is specified, the pattern will be modified as follows:
            ``pet_linear/sub-*_ses-*_trc-{tracer}_rec-{reconstruction}_space-MNI152NLin2009cSym_{desc-Crop}_res-1x1x1_suvr-{suvr_reference_region}_pet.nii*``
    """

    suvr_reference_region: SUVRReferenceRegion = SUVRReferenceRegion.PONS

    @computed_field
    @property
    def name(self) -> str:
        """The preprocessing method."""
        return PreprocessingMethod.PET_LINEAR.value

    def _get_description(self) -> str:
        """
        Constructs a description depending on the preprocessing parameters.
        """
        description = f"PET images with tracer '{self.tracer}'"
        if self.reconstruction:
            description += f" and reconstruction method '{self.reconstruction}'"
        description += (
            f", registered to MNI152NLin2009cSym space using Clinica's '{self.name}' pipeline "
            f"with SUVR reference region '{self.suvr_reference_region}'"
        )
        if not self.use_uncropped_image:
            description += (
                ", and cropped (matrix size 169×208×179, 1 mm isotropic voxels)"
            )
        return description

    def _get_file_pattern(self):
        """
        Constructs the file pattern depending on the parameters of 'pet-linear'.
        """
        desc_crop = "" if self.use_uncropped_image else "_desc-Crop"
        rec = f"_rec-{self.reconstruction}" if self.reconstruction else ""
        return f"sub-*_ses-*_trc-{self.tracer}{rec}_space-MNI152NLin2009cSym{desc_crop}_res-1x1x1_suvr-{self.suvr_reference_region}_{self.modality}.nii*"

    def _get_file_name(self) -> str:
        """
        Builds a suffix for files saving
        information on this preprocessing.
        """
        return (
            f"pet-linear_{self.tracer}_{self.suvr_reference_region}{'_' + self.reconstruction if self.reconstruction else ''}"
            f"{'' if self.use_uncropped_image else '_cropped'}"
        )
