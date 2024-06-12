import tarfile
from logging import getLogger
from pathlib import Path

import click
import nibabel as nib
import pandas as pd
from joblib import Parallel, delayed

from clinicadl.caps_dataset.caps_dataset_config import CapsDatasetConfig
from clinicadl.commandline import arguments
from clinicadl.commandline.modules_options import (
    data,
    dataloader,
    extraction,
    preprocessing,
)
from clinicadl.commandline.pipelines.generate.trivial import options as trivial
from clinicadl.generate.generate_config import GenerateTrivialConfig
from clinicadl.generate.generate_utils import (
    find_file_type,
    im_loss_roi_gaussian_distribution,
    load_and_check_tsv,
    write_missing_mods,
)
from clinicadl.tsvtools.tsvtools_utils import extract_baseline
from clinicadl.utils.clinica_utils import clinicadl_file_reader
from clinicadl.utils.enum import ExtractionMethod
from clinicadl.utils.maps_manager.iotools import commandline_to_json
from clinicadl.utils.read_utils import get_mask_path

logger = getLogger("clinicadl.generate.trivial")


@click.command(name="trivial", no_args_is_help=True)
@arguments.caps_directory
@arguments.generated_caps_directory
@preprocessing.preprocessing
@data.participants_tsv
@data.n_subjects
@dataloader.n_proc
@extraction.use_uncropped_image
@preprocessing.tracer
@preprocessing.suvr_reference_region
@trivial.atrophy_percent
@data.mask_path
def cli(generated_caps_directory, **kwargs):
    """Generation of a trivial dataset"""

    caps_config = CapsDatasetConfig.from_preprocessing_and_extraction_method(
        extraction_type=ExtractionMethod.IMAGE,
        preprocessing_type=kwargs["preprocessing"],
        **kwargs,
    )

    generate_config = GenerateTrivialConfig(**kwargs)

    # TODO: put more information in json file
    commandline_to_json(
        {
            "output_dir": generated_caps_directory,
            "caps_dir": caps_config.data.caps_directory,
            "preprocessing": caps_config.preprocessing.preprocessing.value,
        }
    )
    # Read DataFrame
    data_df = load_and_check_tsv(
        caps_config.data.data_tsv,
        caps_config.data.caps_dict,
        generated_caps_directory,
    )
    data_df = extract_baseline(data_df)
    if caps_config.data.n_subjects > len(data_df):
        raise IndexError(
            f"The number of subjects {caps_config.data.n_subjects} cannot be higher "
            f"than the number of subjects in the baseline dataset of size {len(data_df)}"
        )

    # Create subjects dir
    (generated_caps_directory / "subjects").mkdir(parents=True, exist_ok=True)

    # Find appropriate preprocessing file type
    file_type = find_file_type(caps_config)

    # Output tsv file
    diagnosis_list = ["AD", "CN"]

    def create_trivial_image(subject_id: int) -> pd.DataFrame:
        data_idx = subject_id // 2
        label = subject_id % 2

        participant_id = data_df.at[data_idx, "participant_id"]
        session_id = data_df.at[data_idx, "session_id"]
        cohort = data_df.at[data_idx, "cohort"]
        image_path = Path(
            clinicadl_file_reader(
                [participant_id],
                [session_id],
                caps_config.data.caps_dict[cohort],
                file_type,
            )[0][0]
        )

        from clinicadl.utils.read_utils import get_info_from_filename

        _, _, filename_pattern, file_suffix = get_info_from_filename(image_path)

        # input_filename = image_path.name
        # filename_pattern = "_".join(input_filename.split("_")[2::])

        trivial_image_nii_dir = (
            generated_caps_directory
            / "subjects"
            / f"sub-TRIV{subject_id}"
            / session_id
            / caps_config.preprocessing.preprocessing.value
        )
        trivial_image_nii_dir.mkdir(parents=True, exist_ok=True)

        if caps_config.data.mask_path is None:
            caps_config.data.mask_path = get_mask_path()
        path_to_mask = caps_config.data.mask_path / f"mask-{label + 1}.nii"
        if path_to_mask.is_file():
            atlas_to_mask = nib.loadsave.load(path_to_mask).get_fdata()
        else:
            raise ValueError("masks need to be named mask-1.nii and mask-2.nii")

        image_nii = nib.loadsave.load(image_path)
        image = image_nii.get_fdata()

        # Create atrophied image
        trivial_image = im_loss_roi_gaussian_distribution(
            image, atlas_to_mask, generate_config.atrophy_percent
        )
        trivial_image_nii = nib.nifti1.Nifti1Image(
            trivial_image, affine=image_nii.affine
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

    results_df = Parallel(n_jobs=caps_config.dataloader.n_proc)(
        delayed(create_trivial_image)(subject_id)
        for subject_id in range(2 * caps_config.data.n_subjects)
    )
    output_df = pd.DataFrame()
    for result in results_df:
        output_df = pd.concat([result, output_df])

    output_df.to_csv(generated_caps_directory / "data.tsv", sep="\t", index=False)
    write_missing_mods(generated_caps_directory, output_df)
    logger.info(f"Trivial dataset was generated at {generated_caps_directory}")


if __name__ == "__main__":
    cli()
