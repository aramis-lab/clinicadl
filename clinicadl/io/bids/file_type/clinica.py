import os
from enum import Enum
from typing import Any, Optional

from .base import BidsFileType

__all__ = ["T1Linear", "FlairLinear", "PetLinear", "DwiDti"]

MNI = "MNI152NLin2009cSym"
ISOTROPIC_1 = "1x1x1"
CROP = "Crop"

## t1-linear ##


class T1Linear(BidsFileType):
    """
    :py:class:`~clinicadl.io.bids.BidsFileType` to select T1-weighted MRI images
    preprocessed with :clinica:`Clinica t1-linear <Pipelines/T1_Linear/>` pipeline.

    Parameters
    ----------
    use_uncropped_image : bool, default=False
        Whether to use the uncropped images returned by ``Clinica``:

        - if ``use_uncropped_image=True``: only the files that match the pattern
          ``t1_linear/sub-*_ses-*_space-MNI152NLin2009cSym_res-1x1x1_T1w.nii*``
          will be selected.
        - else: only the files that match the pattern
          ``t1_linear/sub-*_ses-*_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w.nii*``
          will be selected.
    """

    def __init__(self, use_uncropped_image: bool = False):
        entities = {
            "space": MNI,
            "res": ISOTROPIC_1,
        }

        description = _get_decription_linear_pipeline(
            modality="T1 weighted",
            pipeline_name="t1-linear",
            use_uncropped_image=use_uncropped_image,
        )

        without_entities = {}
        if not use_uncropped_image:
            entities = _insert_crop(entities)
        else:
            without_entities["desc"] = CROP

        super().__init__(
            data_type="t1_linear",
            suffix="T1w",
            with_entities=entities,
            description=description,
            without_entities=without_entities,
        )


## flair-linear ##


class FlairLinear(BidsFileType):
    """
    :py:class:`~clinicadl.io.bids.BidsFileType` to select Fluid-Attenuated Inversion Recovery (FLAIR) MRI images
    preprocessed with :clinica:`Clinica flair-linear <Pipelines/FLAIR_Linear/>` pipeline.

    Parameters
    ----------
    use_uncropped_image : bool, default=False
        Whether to use the uncropped images returned by ``Clinica``:\n
        - if ``use_uncropped_image=True``: only the files that match the pattern
          ``flair_linear/sub-*_ses-*_space-MNI152NLin2009cSym_res-1x1x1_FLAIR.nii*``
          will be selected.
        - else: only the files that match the pattern
          ``flair_linear/sub-*_ses-*_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_FLAIR.nii*``
          will be selected.
    """

    def __init__(self, use_uncropped_image: bool = False):
        entities = {
            "space": MNI,
            "res": ISOTROPIC_1,
        }

        description = _get_decription_linear_pipeline(
            modality="FLAIR",
            pipeline_name="flair-linear",
            use_uncropped_image=use_uncropped_image,
        )

        without_entities = {}
        if not use_uncropped_image:
            entities = _insert_crop(entities)
        else:
            without_entities["desc"] = CROP

        super().__init__(
            data_type="flair_linear",
            suffix="FLAIR",
            with_entities=entities,
            description=description,
            without_entities=without_entities,
        )


## pet-linear ##


class Tracer(str, Enum):
    """BIDS label for PET tracers.

    Follows the convention proposed in the PET section of the BIDS specification.

    See: https://bids-specification.readthedocs.io/en/stable/04-modality-specific-files/09-positron-emission-tomography.html
    """

    PIB = "11CPIB"
    AV1451 = "18FAV1451"
    AV45 = "18FAV45"
    FBB = "18FFBB"
    FDG = "18FFDG"
    FMM = "18FFMM"


class ReconstructionMethod(str, Enum):
    """BIDS label for PET reconstruction methods.

    Follows the convention proposed in the PET section of the BIDS specification.

    See: https://bids-specification.readthedocs.io/en/stable/04-modality-specific-files/09-positron-emission-tomography.html#pet-recording-data

    For ADNI specific reconstruction methods, see:

    https://adni.loni.usc.edu/data-samples/adni-data/neuroimaging/pet/
    """

    # Reconstruction methods defined in the BIDS specifications
    STATIC = "nacstat"
    DYNAMIC = "nacdyn"
    STATIC_ATTENUATION_CORRECTION = "acstat"
    DYNAMIC_ATTENUATION_CORRECTION = "acdyn"

    # ADNI specific reconstruction methods
    CO_REGISTERED_DYNAMIC = "coregdyn"  # Corresponds to ADNI processing steps 1
    CO_REGISTERED_AVERAGED = "coregavg"  # Corresponds to ADNI processing steps 2
    CO_REGISTERED_STANDARDIZED = "coregstd"  # Corresponds to ADNI processing steps 3
    COREGISTERED_ISOTROPIC = "coregiso"  # Corresponds to ADNI processing steps 4


class SUVRReferenceRegion(str, Enum):
    """Supported SUVR reference region in Clinica."""

    PONS = "pons"
    CEREBELLUM_PONS = "cerebellumPons"
    PONS2 = "pons2"
    CEREBELLUM_PONS2 = "cerebellumPons2"


class PetLinear(BidsFileType):
    """
    :py:class:`~clinicadl.io.bids.BidsFileType` to select Positron Emission Tomography (PET) images
    preprocessed with :clinica:`Clinica pet-linear <Pipelines/PET_Linear/>` pipeline.

    Parameters
    ----------
    tracer : str | Tracer
        The radioactive tracer used for acquisition, among ``11CPIB``, ``18FAV1451``, ``18FAV45``, ``18FFBB``,
        ``18FFDG`` and ``18FFMM``.

    suvr_reference_region : str | SUVRReferenceRegion
        The reference region used to compute SUVR, among ``pons``, ``cerebellumPons``, ``pons2`` and ``cerebellumPons2``.

    reconstruction : Optional[ReconstructionMethod], default=None
        The method used to reconstruct the image, among ``nacstat``, ``nacdyn``, ``acstat``, ``acdyn``, ``coregdyn``,
        ``coregavg``, ``coregstd`` and ``coregiso``. Leave to ``None`` if not specified.

    use_uncropped_image : bool, default=False
        Whether to use the uncropped images returned by ``Clinica``:\n
        - if ``use_uncropped_image=True``: only the files that match the pattern
          ``pet_linear/sub-*_ses-*_trc-{tracer}_space-MNI152NLin2009cSym_res-1x1x1_suvr-{suvr_reference_region}_pet.nii*``
          will be selected.
        - else: only the files that match the pattern
          ``pet_linear/sub-*_ses-*_trc-{tracer}_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_suvr-{suvr_reference_region}_pet.nii*``
          will be selected.

        .. note::
            If ``reconstruction`` is specified, the pattern will be modified as follows:
            ``pet_linear/sub-*_ses-*_trc-{tracer}_rec-{reconstruction}_space-MNI152NLin2009cSym_(desc-Crop)_res-1x1x1_suvr-{suvr_reference_region}_pet.nii*``
    """

    def __init__(
        self,
        tracer: str | Tracer,
        suvr_reference_region: str | SUVRReferenceRegion,
        reconstruction: Optional[str | ReconstructionMethod] = None,
        use_uncropped_image: bool = False,
    ):
        tracer = Tracer(tracer).value
        suvr = SUVRReferenceRegion(suvr_reference_region).value
        if reconstruction:
            reconstruction = ReconstructionMethod(reconstruction).value

        entities = {
            "trc": tracer,
            "space": MNI,
            "res": ISOTROPIC_1,
            "suvr": suvr,
        }

        without_entities = {}
        if not use_uncropped_image:
            entities = _insert_crop(entities)
        else:
            without_entities["desc"] = CROP

        if reconstruction:
            entities["rec"] = reconstruction

        description = _get_decription_linear_pipeline(
            modality="PET",
            pipeline_name="pet-linear",
            use_uncropped_image=use_uncropped_image,
            tracer=tracer,
            suvr=suvr,
            reconstruction=reconstruction,
        )

        super().__init__(
            data_type="pet_linear",
            suffix="pet",
            with_entities=entities,
            description=description,
            without_entities=without_entities,
        )


## dwi-dti ##


class DTIMeasure(str, Enum):
    """Possible DTI measures."""

    FRACTIONAL_ANISOTROPY = "FA"
    MEAN_DIFFUSIVITY = "MD"
    AXIAL_DIFFUSIVITY = "AD"
    RADIAL_DIFFUSIVITY = "RD"


class DTISpace(str, Enum):
    """Possible DTI spaces."""

    NATIVE = "native"
    NORMALIZED = "normalized"


class DwiDti(BidsFileType):
    """
    :py:class:`~clinicadl.io.bids.BidsFileType` to select Diffusion-Weighted MRI (DWI) images
    preprocessed with :clinica:`Clinica dwi-dti <Pipelines/DWI_DTI/>` pipeline.

    Parameters
    ----------
    measure : str | DTIMeasure
        The DTI-based measure to use, among ``FA`` (fractional anisotropy),
        ``MD`` (mean diffusivity), ``AD`` (axial diffusivity) and ``RD`` (radial diffusivity).
    space : str | DTISpace
        Either ``native`` (the data in the native space) or ``normalized`` (the data in
        MNI152Lin standard space):\n
        - with ``native``: only the files that match the pattern
          ``dwi/dti_based_processing/native_space/sub-*_ses-*_space-{b0|T1w}_{measure}.nii*``
          will be selected.
        - with ``normalized``: only the files that match the pattern
          ``dwi/dti_based_processing/normalized_space/sub-*_ses-*_space-MNI152Lin_res-1x1x1_{measure}.nii*``
          will be selected.
    """

    def __init__(
        self,
        measure: str | DTIMeasure,
        space: str | DTISpace,
    ):
        space = DTISpace(space).value
        measure = DTIMeasure(measure).value

        if space == DTISpace.NORMALIZED:
            data_type = "normalized_space"
            with_entities = {"space": "MNI152Lin", "res": "1x1x1"}
        else:
            data_type = "native_space"
            with_entities = {"space": r"\b(b0|T1w)\b"}

        super().__init__(
            data_type=os.path.join("dwi", "dti_based_processing", data_type),
            suffix=measure,
            with_entities=with_entities,
            description=f"DTI {measure} images in {space} space, preprocessed with Clinica's 'dwi-dti' pipeline.",
        )


####


def _get_decription_linear_pipeline(
    modality: str,
    pipeline_name: str,
    use_uncropped_image: bool,
    tracer: Optional[str] = None,
    suvr: Optional[str] = None,
    reconstruction: Optional[str] = None,
) -> str:
    """
    Writes the description of Clinica's linear pipelines.
    """
    description = f"{modality} images "

    if tracer:
        description += f"with tracer '{tracer}' "

    if reconstruction:
        description += f"and reconstruction method '{reconstruction}', "

    description += (
        f"registered to {MNI} space using Clinica's '{pipeline_name}' pipeline"
    )

    if suvr:
        description += f" with SUVR reference region '{suvr}'"

    if not use_uncropped_image:
        description += ", and cropped (matrix size 169×208×179, 1 mm isotropic voxels)"

    return description + "."


def _insert_crop(input_dict: dict[str, Any]) -> dict[str, Any]:
    """
    Insert crop entity just after 'space' entity.
    """
    items = list(input_dict.items())
    items.insert(list(input_dict.keys()).index("space") + 1, ("desc", CROP))

    return dict(items)
