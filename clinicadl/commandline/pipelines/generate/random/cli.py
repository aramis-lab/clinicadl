from logging import getLogger
from pathlib import Path

import click
import nibabel as nib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from clinicadl.caps_dataset.caps_dataset_config import create_caps_dataset_config
from clinicadl.caps_dataset.data import CapsDataset
from clinicadl.commandline import arguments
from clinicadl.commandline.modules_options import (
    data,
    dataloader,
    modality,
    preprocessing,
)
from clinicadl.commandline.pipelines.generate.random import options as random
from clinicadl.generate.generate_config import GenerateRandomConfig
from clinicadl.generate.generate_utils import (
    find_file_type,
    load_and_check_tsv,
    write_missing_mods,
)
from clinicadl.tsvtools.tsvtools_utils import extract_baseline
from clinicadl.utils.clinica_utils import clinicadl_file_reader
from clinicadl.utils.enum import ExtractionMethod, GenerateType, Preprocessing
from clinicadl.utils.maps_manager.iotools import commandline_to_json

logger = getLogger("clinicadl.generate.random")


@click.command(name="random", no_args_is_help=True)
@arguments.caps_directory
@arguments.generated_caps_directory
@preprocessing.preprocessing
@data.participants_tsv
@data.n_subjects
@dataloader.n_proc
@preprocessing.use_uncropped_image
@modality.tracer
@modality.suvr_reference_region
@random.mean
@random.sigma
def cli(generated_caps_directory, n_proc, **kwargs):
    """Addition of random gaussian noise to brain images.
    CAPS_DIRECTORY is the CAPS folder from where input brain images will be loaded.
    GENERATED_CAPS_DIRECTORY is a CAPS folder where the random dataset will be saved.
    """

    caps_config = create_caps_dataset_config(
        extract=ExtractionMethod.IMAGE,
        preprocessing=Preprocessing(kwargs["preprocessing"]),
    )(**kwargs)
    generate_config = GenerateRandomConfig(**kwargs)

    # TODO: put more information in json file
    commandline_to_json(
        {
            "output_dir": generated_caps_directory,
            "caps_dir": caps_config.data.caps_directory,
            "preprocessing": caps_config.preprocessing.preprocessing.value,
            "n_subjects": caps_config.data.n_subjects,
            "n_proc": n_proc,
            "mean": generate_config.mean,
            "sigma": generate_config.sigma,
        }
    )
    SESSION_ID = "ses-M000"
    AGE_BL_DEFAULT = 60
    SEX_DEFAULT = "F"

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

    # Retrieve image of first subject
    participant_id = data_df.at[0, "participant_id"]
    session_id = data_df.at[0, "session_id"]
    cohort = data_df.at[0, "cohort"]
    image_paths = clinicadl_file_reader(
        [participant_id], [session_id], caps_config.data.caps_dict[cohort], file_type
    )
    image_nii = nib.loadsave.load(image_paths[0][0])
    # assert isinstance(image_nii, nib.nifti1.Nifti1Image)
    image = image_nii.get_fdata()
    output_df = pd.DataFrame(
        {
            "participant_id": [
                f"sub-RAND{i}" for i in range(2 * caps_config.data.n_subjects)
            ],
            "session_id": [SESSION_ID] * 2 * caps_config.data.n_subjects,
            "diagnosis": ["AD"] * caps_config.data.n_subjects
            + ["CN"] * caps_config.data.n_subjects,
            "age_bl": AGE_BL_DEFAULT,
            "sex": SEX_DEFAULT,
        }
    )
    output_df.to_csv(generated_caps_directory / "data.tsv", sep="\t", index=False)
    input_filename = Path(image_paths[0][0]).name
    filename_pattern = "_".join(input_filename.split("_")[2:])

    def create_random_image(subject_id: int) -> None:
        gauss = np.random.normal(
            generate_config.mean, generate_config.sigma, image.shape
        )
        # TODO: warning: use np.random.Generator(PCG64) puis .standard_random()
        participant_id = f"sub-RAND{subject_id}"
        noisy_image = image + gauss
        noisy_image_nii = nib.nifti1.Nifti1Image(
            noisy_image, header=image_nii.header, affine=image_nii.affine
        )
        noisy_image_nii_path = (
            generated_caps_directory
            / "subjects"
            / participant_id
            / SESSION_ID
            / "t1_linear"
        )
        noisy_image_nii_filename = f"{participant_id}_{SESSION_ID}_{filename_pattern}"
        noisy_image_nii_path.mkdir(parents=True, exist_ok=True)
        nib.loadsave.save(
            noisy_image_nii, noisy_image_nii_path / noisy_image_nii_filename
        )

    Parallel(n_jobs=n_proc)(
        delayed(create_random_image)(subject_id)
        for subject_id in range(2 * caps_config.data.n_subjects)
    )
    write_missing_mods(generated_caps_directory, output_df)
    logger.info(f"Random dataset was generated at {generated_caps_directory}")


if __name__ == "__main__":
    cli()
