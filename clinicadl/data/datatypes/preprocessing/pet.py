import os
import re
from enum import Enum
from typing import Pattern

from ..modalities import PET
from .base import _LinearPreprocessing


class SUVRReferenceRegion(str, Enum):
    """Supported SUVR reference region in Clinica."""

    PONS = "pons"
    CEREBELLUM_PONS = "cerebellumPons"
    PONS2 = "pons2"
    CEREBELLUM_PONS2 = "cerebellumPons2"


class PETLinear(PET, _LinearPreprocessing):
    """
    :py:class:`DataType <clinicadl.data.datatypes.DataType>` to handle Positron Emission Tomography (PET) images
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

    suvr_reference_region: SUVRReferenceRegion

    @property
    def _pipeline_name(self) -> str:
        return "pet-linear"

    @property
    def _filename(self) -> str:
        return (
            f"{self._pipeline_name}_{self.tracer}_{self.suvr_reference_region}{'_' + self.reconstruction if self.reconstruction else ''}"
            f"{'' if self.use_uncropped_image else '_cropped'}"
        )

    def _get_description(self) -> str:
        description = f"PET images with tracer '{self.tracer}'"
        if self.reconstruction:
            description += f" and reconstruction method '{self.reconstruction}'"
        description += (
            f", registered to MNI152NLin2009cSym space using Clinica's '{self._pipeline_name}' pipeline "
            f"with SUVR reference region '{self.suvr_reference_region}'"
        )
        if not self.use_uncropped_image:
            description += (
                ", and cropped (matrix size 169×208×179, 1 mm isotropic voxels)"
            )
        return description

    def _get_pattern(self) -> Pattern:
        desc_crop = "" if self.use_uncropped_image else "_desc-Crop"
        rec = f"_rec-{self.reconstruction}" if self.reconstruction else ""
        file_pattern = f"sub-.*_ses-.*_trc-{self.tracer}{rec}_space-MNI152NLin2009cSym{desc_crop}_res-1x1x1_suvr-{self.suvr_reference_region}_{self._modality}.nii.*"
        pattern = os.path.join(self._pipeline_name.replace("-", "_"), file_pattern)

        return re.compile(pattern)
