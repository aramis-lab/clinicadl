from logging import getLogger
from pathlib import Path

import click
import nibabel as nib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from clinicadl.generate import generate_param
from clinicadl.generate.generate_config import GenerateRandomConfig
from clinicadl.utils.caps_dataset.data import CapsDataset
from clinicadl.utils.clinica_utils import clinicadl_file_reader
from clinicadl.utils.maps_manager.iotools import commandline_to_json

from .generate_utils import (
    find_file_type,
    load_and_check_tsv,
    write_missing_mods,
)

logger = getLogger("clinicadl.generate.random")


@click.command(name="random", no_args_is_help=True)
@generate_param.argument.caps_directory
@generate_param.argument.generated_caps_directory
@generate_param.option.preprocessing
@generate_param.option.participants_tsv
@generate_param.option.n_subjects
@generate_param.option.n_proc
@generate_param.option.use_uncropped_image
@generate_param.option.tracer
@generate_param.option.suvr_reference_region
@generate_param.option_random.mean
@generate_param.option_random.sigma
def cli(caps_directory, generated_caps_directory, **kwargs):
    """Addition of random gaussian noise to brain images.

    CAPS_DIRECTORY is the CAPS folder from where input brain images will be loaded.

    GENERATED_CAPS_DIRECTORY is a CAPS folder where the random dataset will be saved.
    """
    random_config = GenerateRandomConfig(
        caps_directory=caps_directory,
        generated_caps_directory=generated_caps_directory,
        preprocessing_cls=kwargs["preprocessing"],
        tracer_cls=kwargs["tracer"],
        suvr_reference_region_cls=kwargs["suvr_reference_region"],
        **kwargs,
    )

    commandline_to_json(
        {
            "output_dir": random_config.generated_caps_directory,
            "caps_dir": caps_directory,
            "preprocessing": random_config.preprocessing,
            "n_subjects": random_config.n_subjects,
            "n_proc": random_config.n_proc,
            "mean": random_config.mean,
            "sigma": random_config.sigma,
        }
    )

    SESSION_ID = "ses-M000"
    AGE_BL_DEFAULT = 60
    SEX_DEFAULT = "F"
    multi_cohort = False  # ??? hard coded ?

    # Transform caps_directory in dict
    caps_dict = CapsDataset.create_caps_dict(caps_directory, multi_cohort=multi_cohort)

    # Read DataFrame
    data_df = load_and_check_tsv(
        random_config.participants_list,
        caps_dict,
        random_config.generated_caps_directory,
    )

    # Create subjects dir
    (random_config.generated_caps_directory / "subjects").mkdir(
        parents=True, exist_ok=True
    )

    # Retrieve image of first subject
    participant_id = data_df.loc[0, "participant_id"]
    session_id = data_df.loc[0, "session_id"]
    cohort = data_df.loc[0, "cohort"]

    # Find appropriate preprocessing file type
    file_type = find_file_type(
        random_config.preprocessing,
        random_config.use_uncropped_image,
        random_config.tracer,
        random_config.suvr_reference_region,
    )

    image_paths = clinicadl_file_reader(
        [participant_id], [session_id], caps_dict[cohort], file_type
    )
    image_nii = nib.load(image_paths[0][0])
    image = image_nii.get_fdata()

    output_df = pd.DataFrame(
        {
            "participant_id": [
                f"sub-RAND{i}" for i in range(2 * random_config.n_subjects)
            ],
            "session_id": [SESSION_ID] * 2 * random_config.n_subjects,
            "diagnosis": ["AD"] * random_config.n_subjects
            + ["CN"] * random_config.n_subjects,
            "age_bl": AGE_BL_DEFAULT,
            "sex": SEX_DEFAULT,
        }
    )

    output_df.to_csv(
        random_config.generated_caps_directory / "data.tsv", sep="\t", index=False
    )

    input_filename = Path(image_paths[0][0]).name
    filename_pattern = "_".join(input_filename.split("_")[2:])

    def create_random_image(subject_id: int) -> None:
        gauss = np.random.normal(random_config.mean, random_config.sigma, image.shape)
        participant_id = f"sub-RAND{subject_id}"
        noisy_image = image + gauss
        noisy_image_nii = nib.Nifti1Image(
            noisy_image, header=image_nii.header, affine=image_nii.affine
        )
        noisy_image_nii_path = (
            random_config.generated_caps_directory
            / "subjects"
            / participant_id
            / SESSION_ID
            / "t1_linear"
        )

        noisy_image_nii_filename = f"{participant_id}_{SESSION_ID}_{filename_pattern}"
        noisy_image_nii_path.mkdir(parents=True, exist_ok=True)
        nib.save(noisy_image_nii, noisy_image_nii_path / noisy_image_nii_filename)

    Parallel(n_jobs=random_config.n_proc)(
        delayed(create_random_image)(subject_id)
        for subject_id in range(2 * random_config.n_subjects)
    )

    write_missing_mods(random_config.generated_caps_directory, output_df)
    logger.info(
        f"Random dataset was generated at {random_config.generated_caps_directory}"
    )


if __name__ == "__main__":
    cli()
