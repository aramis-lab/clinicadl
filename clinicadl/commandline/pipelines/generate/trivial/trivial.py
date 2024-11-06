from logging import getLogger
from pathlib import Path
from typing import Optional

import nibabel as nib
import pandas as pd
from joblib import Parallel, delayed

from clinicadl.dataset.config import PreprocessingConfig
from clinicadl.generate.generate_utils import (
    im_loss_roi_gaussian_distribution,
    load_and_check_tsv_v2,
    write_missing_mods,
)
from clinicadl.tsvtools.tsvtools_utils import extract_baseline
from clinicadl.utils.iotools.clinica_utils import clinicadl_file_reader
from clinicadl.utils.iotools.read_utils import get_info_from_filename, get_mask_path


def generate_trivial_data(
    preprocessing: PreprocessingConfig,
    caps_directory: Path,
    generated_caps_directory: Path,
    tsv_path: Optional[Path] = None,
    n_subjects: int = 10,
    atrophy_percent: float = 0.4,
    n_proc: int = 2,
    mask_path: Optional[Path] = None,
):
    # Read DataFrame
    data_df = load_and_check_tsv_v2(
        caps_directory=caps_directory,
        output_path=generated_caps_directory,
        tsv_path=tsv_path,
    )
    data_df = extract_baseline(data_df)
    if n_subjects > len(data_df):
        raise IndexError(
            f"The number of subjects {n_subjects} cannot be higher "
            f"than the number of subjects in the baseline dataset of size {len(data_df)}"
        )
    # Create subjects dir
    (generated_caps_directory / "subjects").mkdir(parents=True, exist_ok=True)

    # Find appropriate preprocessing file type
    file_type = preprocessing.get_filetype()

    # Output tsv file
    diagnosis_list = ["AD", "CN"]

    # Initialize logger
    logger = getLogger("clinicadl.generate.trivial")

    if mask_path is None:
        mask_path = get_mask_path()

    def create_trivial_image(subject_id: int) -> pd.DataFrame:
        data_idx = subject_id // 2
        label = subject_id % 2

        participant_id = data_df.at[data_idx, "participant_id"]
        session_id = data_df.at[data_idx, "session_id"]
        image_path = Path(
            clinicadl_file_reader(
                [participant_id],
                [session_id],
                caps_directory,
                file_type,
            )[0][0]
        )

        _, _, filename_pattern, file_suffix = get_info_from_filename(image_path)

        trivial_image_nii_dir = (
            generated_caps_directory
            / "subjects"
            / f"sub-TRIV{subject_id}"
            / session_id
            / preprocessing.preprocessing.value
        )
        trivial_image_nii_dir.mkdir(parents=True, exist_ok=True)

        path_to_mask = mask_path / f"mask-{label + 1}.nii"

        if path_to_mask.is_file():
            atlas_to_mask = nib.loadsave.load(path_to_mask).get_fdata()  # type: ignore
        else:
            raise ValueError("masks need to be named mask-1.nii and mask-2.nii")

        image_nii = nib.loadsave.load(image_path)
        image = image_nii.get_fdata()  # type: ignore

        # Create atrophied image
        trivial_image = im_loss_roi_gaussian_distribution(
            image, atlas_to_mask, atrophy_percent
        )
        trivial_image_nii = nib.nifti1.Nifti1Image(
            trivial_image,
            affine=image_nii.affine,  # type: ignore
        )
        trivial_image_nii_filename = (
            f"sub-TRIV{subject_id}_{session_id}_{filename_pattern + file_suffix}"
        )

        trivial_image_nii.to_filename(
            trivial_image_nii_dir / trivial_image_nii_filename
        )

        # Append row to output tsv
        row = [f"sub-TRIV{subject_id}", session_id, diagnosis_list[label], 60, "F"]
        columns = ["participant_id", "session_id", "diagnosis", "age_bl", "sex"]
        row_df = pd.DataFrame([row], columns=columns)

        return row_df

    results_df = Parallel(n_jobs=n_proc)(
        delayed(create_trivial_image)(subject_id)
        for subject_id in range(2 * n_subjects)
    )
    output_df = pd.DataFrame()
    for result in results_df:
        output_df = pd.concat([result, output_df])

    output_df.to_csv(generated_caps_directory / "data.tsv", sep="\t", index=False)
    write_missing_mods(generated_caps_directory, output_df)
    logger.info(f"Trivial dataset was generated at {generated_caps_directory}")

    return generated_caps_directory
