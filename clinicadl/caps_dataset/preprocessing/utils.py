from typing import Optional

from clinicadl.caps_dataset.preprocessing import config as preprocessing_config
from clinicadl.utils.clinica_utils import FileType
from clinicadl.utils.enum import (
    LinearModality,
    Preprocessing,
    Tracer,
)
from clinicadl.utils.exceptions import (
    ClinicaDLArgumentError,
)


def bids_nii(
    config: preprocessing_config.PreprocessingConfig,
    reconstruction: Optional[str] = None,
) -> FileType:
    """Return the query dict required to capture PET scans.

    Parameters
    ----------
    tracer : Tracer, optional
        If specified, the query will only match PET scans acquired
        with the requested tracer.
        If None, the query will match all PET sans independently of
        the tracer used.

    reconstruction : ReconstructionMethod, optional
        If specified, the query will only match PET scans reconstructed
        with the specified method.
        If None, the query will match all PET scans independently of the
        reconstruction method used.

    Returns
    -------
    dict :
        The query dictionary to get PET scans.
    """

    if config.preprocessing not in Preprocessing:
        raise ClinicaDLArgumentError(
            f"ClinicaDL is Unable to read this modality ({config.preprocessing}) of images, please chose one from this list: {list[Preprocessing]}"
        )

    if isinstance(config, preprocessing_config.PETPreprocessingConfig):
        trc = "" if config.tracer is None else f"_trc-{Tracer(config.tracer).value}"
        rec = "" if reconstruction is None else f"_rec-{reconstruction}"
        description = "PET data"

        if config.tracer:
            description += f" with {config.tracer.value} tracer"
        if reconstruction:
            description += f" and reconstruction method {reconstruction}"

        file_type = FileType(
            pattern=f"pet/*{trc}{rec}_pet.nii*", description=description
        )
        return file_type

    elif isinstance(config, preprocessing_config.T1PreprocessingConfig):
        return FileType(pattern="anat/sub-*_ses-*_T1w.nii*", description="T1w MRI")

    elif isinstance(config, preprocessing_config.FlairPreprocessingConfig):
        return FileType(pattern="sub-*_ses-*_flair.nii*", description="FLAIR T2w MRI")

    elif isinstance(config, preprocessing_config.DTIPreprocessingConfig):
        return FileType(pattern="dwi/sub-*_ses-*_dwi.nii*", description="DWI NIfTI")

    else:
        raise ClinicaDLArgumentError("Invalid preprocessing")


def linear_nii(
    config: preprocessing_config.PreprocessingConfig,
) -> FileType:
    if isinstance(config, preprocessing_config.T1PreprocessingConfig):
        needed_pipeline = Preprocessing.T1_LINEAR
        modality = LinearModality.T1W
    elif isinstance(config, preprocessing_config.T2PreprocessingConfig):
        needed_pipeline = Preprocessing.T2_LINEAR
        modality = LinearModality.T2W
    elif isinstance(config, preprocessing_config.FlairPreprocessingConfig):
        needed_pipeline = Preprocessing.FLAIR_LINEAR
        modality = LinearModality.FLAIR
    else:
        raise ClinicaDLArgumentError("Invalid configuration")

    if config.use_uncropped_image:
        desc_crop = ""
    else:
        desc_crop = "_desc-Crop"

    file_type = FileType(
        pattern=f"*space-MNI152NLin2009cSym{desc_crop}_res-1x1x1_{modality.value}.nii.gz",
        description=f"{modality.value} Image registered in MNI152NLin2009cSym space using {needed_pipeline.value} pipeline "
        + (
            ""
            if config.use_uncropped_image
            else "and cropped (matrix size 169×208×179, 1 mm isotropic voxels)"
        ),
        needed_pipeline=needed_pipeline,
    )
    return file_type


def dwi_dti(config: preprocessing_config.DTIPreprocessingConfig) -> FileType:
    """Return the query dict required to capture DWI DTI images.

    Parameters
    ----------
    config: DTIPreprocessingConfig

    Returns
    -------
    FileType :
    """
    if isinstance(config, preprocessing_config.DTIPreprocessingConfig):
        measure = config.dti_measure
        space = config.dti_space
    else:
        raise ClinicaDLArgumentError(
            f"PreprocessingConfig is of type {config} but should be of type{preprocessing_config.DTIPreprocessingConfig}"
        )

    return FileType(
        pattern=f"dwi/dti_based_processing/*/*_space-{space}_{measure.value}.nii.gz",
        description=f"DTI-based {measure.value} in space {space}.",
        needed_pipeline="dwi_dti",
    )


def pet_linear_nii(config: preprocessing_config.PETPreprocessingConfig) -> FileType:
    if not isinstance(config, preprocessing_config.PETPreprocessingConfig):
        raise ClinicaDLArgumentError(
            f"PreprocessingConfig is of type {config} but should be of type{preprocessing_config.PETPreprocessingConfig}"
        )

    if config.use_uncropped_image:
        description = ""
    else:
        description = "_desc-Crop"

    file_type = FileType(
        pattern=f"pet_linear/*_trc-{config.tracer.value}_space-MNI152NLin2009cSym{description}_res-1x1x1_suvr-{config.suvr_reference_region.value}_pet.nii.gz",
        description="",
        needed_pipeline="pet-linear",
    )
    return file_type
