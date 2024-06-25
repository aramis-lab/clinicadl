from logging import getLogger
from pathlib import Path

import click
import nibabel as nib
import pandas as pd
from joblib import Parallel, delayed
from nilearn.image import resample_to_img

from clinicadl.caps_dataset.caps_dataset_config import CapsDatasetConfig
from clinicadl.caps_dataset.caps_dataset_utils import find_file_type
from clinicadl.commandline import arguments
from clinicadl.commandline.modules_options import data, dataloader, preprocessing
from clinicadl.commandline.pipelines.generate.hypometabolic import (
    options as hypometabolic,
)
from clinicadl.generate.generate_config import GenerateHypometabolicConfig
from clinicadl.generate.generate_utils import (
    load_and_check_tsv,
    mask_processing,
    write_missing_mods,
)
from clinicadl.tsvtools.tsvtools_utils import extract_baseline
from clinicadl.utils.clinica_utils import clinicadl_file_reader
from clinicadl.utils.enum import (
    ExtractionMethod,
    Preprocessing,
)
from clinicadl.utils.iotools.read_utils import get_mask_path
from clinicadl.utils.maps_manager.iotools import commandline_to_json

logger = getLogger("clinicadl.generate.hypometabolic")


@click.command(name="hypometabolic", no_args_is_help=True)
@arguments.caps_directory
@arguments.generated_caps_directory
@dataloader.n_proc
@data.participants_tsv
@data.n_subjects
@preprocessing.use_uncropped_image
@hypometabolic.sigma
@hypometabolic.anomaly_degree
@hypometabolic.pathology
def cli(generated_caps_directory, **kwargs):
    """Generation of trivial dataset with addition of synthetic brain atrophy.
    CAPS_DIRECTORY is the CAPS folder from where input brain images will be loaded.
    GENERATED_CAPS_DIRECTORY is a CAPS folder where the trivial dataset will be saved.
    """
    kwargs["preprocessing"] = "pet-linear"
    caps_config = CapsDatasetConfig.from_preprocessing_and_extraction_method(
        extraction=ExtractionMethod.IMAGE,
        preprocessing_type=Preprocessing.PET_LINEAR,
        **kwargs,
    )

    generate_config = GenerateHypometabolicConfig(**kwargs)

    commandline_to_json(
        {
            "output_dir": generated_caps_directory,
            "caps_dir": caps_config.data.caps_directory,
            "preprocessing": caps_config.preprocessing.preprocessing.value,
            "n_subjects": caps_config.data.n_subjects,
            "n_proc": caps_config.dataloader.n_proc,
            "pathology": generate_config.pathology.value,
            "anomaly_degree": generate_config.anomaly_degree,
        }
    )

    # Read DataFrame
    data_df = load_and_check_tsv(
        caps_config.data.data_tsv, caps_config.data.caps_dict, generated_caps_directory
    )
    data_df = extract_baseline(data_df)
    if caps_config.data.n_subjects > len(data_df):
        raise IndexError(
            f"The number of subjects {caps_config.data.n_subjects} cannot be higher "
            f"than the number of subjects in the baseline dataset of size {len(data_df)}"
            f"Please add the '--n_subjects' option and re-run the command."
        )

    # Create subjects dir
    (generated_caps_directory / "subjects").mkdir(parents=True, exist_ok=True)

    # Find appropriate preprocessing file type
    file_type = find_file_type(caps_config)

    mask_path = get_mask_path(generate_config.pathology)

    mask_nii = nib.loadsave.load(mask_path)

    # Output tsv file
    participants = [
        data_df.at[i, "participant_id"] for i in range(caps_config.data.n_subjects)
    ]
    sessions = [data_df.at[i, "session_id"] for i in range(caps_config.data.n_subjects)]
    cohort = caps_config.data.caps_directory

    images_paths = clinicadl_file_reader(
        participants, sessions, cohort, file_type.model_dump()
    )[0]
    image_nii = nib.loadsave.load(images_paths[0])
    mask_resample_nii = resample_to_img(mask_nii, image_nii, interpolation="nearest")
    mask = mask_resample_nii.get_fdata()
    mask = mask_processing(mask, generate_config.anomaly_degree, generate_config.sigma)

    def generate_hypometabolic_image(
        subject_id: int,
    ) -> pd.DataFrame:
        image_path = Path(images_paths[subject_id])
        image_nii = nib.loadsave.load(image_path)
        image = image_nii.get_fdata()
        if image_path.suffix == ".gz":
            input_filename = Path(image_path.stem).stem
        else:
            input_filename = image_path.stem
        input_filename = input_filename.strip("pet")
        hypo_image_nii_dir = (
            generated_caps_directory
            / "subjects"
            / participants[subject_id]
            / sessions[subject_id]
            / caps_config.preprocessing.preprocessing.value
        )
        hypo_image_nii_filename = f"{input_filename}pat-{generate_config.pathology.value}_deg-{int(generate_config.anomaly_degree)}_pet.nii.gz"
        hypo_image_nii_dir.mkdir(parents=True, exist_ok=True)
        # Create atrophied image
        hypo_image = image * mask
        hypo_image_nii = nib.nifti1.Nifti1Image(hypo_image, affine=image_nii.affine)
        hypo_image_nii.to_filename(hypo_image_nii_dir / hypo_image_nii_filename)
        # Append row to output tsv
        row = [
            participants[subject_id],
            sessions[subject_id],
            generate_config.pathology.value,
            generate_config.anomaly_degree,
        ]

        columns = ["participant_id", "session_id", "pathology", "percentage"]
        row_df = pd.DataFrame([row], columns=columns)
        return row_df

    results_list = Parallel(n_jobs=caps_config.dataloader.n_proc)(
        delayed(generate_hypometabolic_image)(subject_id)
        for subject_id in range(caps_config.data.n_subjects)
    )
    output_df = pd.DataFrame()
    for result_df in results_list:
        output_df = pd.concat([result_df, output_df])
    output_df.to_csv(generated_caps_directory / "data.tsv", sep="\t", index=False)
    write_missing_mods(generated_caps_directory, output_df)
    logger.info(
        f"Hypometabolic dataset was generated, with {generate_config.anomaly_degree} % of "
        f"dementia {generate_config.pathology.value} at {generated_caps_directory}."
    )


if __name__ == "__main__":
    cli()
