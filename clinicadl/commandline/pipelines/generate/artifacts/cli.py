from logging import getLogger
from pathlib import Path

import click
import pandas as pd
import torchio as tio
from joblib import Parallel, delayed

from clinicadl.caps_dataset.caps_dataset_config import (
    CapsDatasetConfig,
)
from clinicadl.caps_dataset.caps_dataset_utils import find_file_type
from clinicadl.commandline import arguments
from clinicadl.commandline.modules_options import (
    data,
    dataloader,
    extraction,
    preprocessing,
)
from clinicadl.commandline.pipelines.generate.artifacts import options as artifacts
from clinicadl.generate.generate_config import GenerateArtifactsConfig
from clinicadl.generate.generate_utils import (
    load_and_check_tsv,
    write_missing_mods,
)
from clinicadl.utils.clinica_utils import clinicadl_file_reader
from clinicadl.utils.enum import ExtractionMethod
from clinicadl.utils.maps_manager.iotools import commandline_to_json

logger = getLogger("clinicadl.generate.artifacts")


@click.command(name="artifacts", no_args_is_help=True)
@arguments.caps_directory
@arguments.generated_caps_directory
@dataloader.n_proc
@preprocessing.preprocessing
@preprocessing.use_uncropped_image
@data.participants_tsv
@preprocessing.tracer
@preprocessing.suvr_reference_region
@artifacts.contrast
@artifacts.motion
@artifacts.noise_std
@artifacts.noise
@artifacts.num_transforms
@artifacts.translation
@artifacts.rotation
@artifacts.gamma
def cli(generated_caps_directory, **kwargs):
    """
    Addition of artifacts (noise, motion or contrast) to brain images

    """

    caps_config = CapsDatasetConfig.from_preprocessing_and_extraction_method(
        extraction=ExtractionMethod.IMAGE,
        preprocessing_type=kwargs["preprocessing"],
        **kwargs,
    )

    generate_config = GenerateArtifactsConfig(**kwargs)

    # TODO: creat function for API mode

    # generated_caps_config = generate(generate_config, caps_config, generated_caps_directory)

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
        caps_config.data.data_tsv, caps_config.data.caps_dict, generated_caps_directory
    )
    # data_df = extract_baseline(data_df)
    # if caps_config.n_subjects > len(data_df):
    #     raise IndexError(
    #         f"The number of subjects {caps_config.n_subjects} cannot be higher "
    #         f"than the number of subjects in the baseline dataset of size {len(data_df)}"
    #     )

    # Create subjects dir
    (generated_caps_directory / "subjects").mkdir(parents=True, exist_ok=True)

    # Find appropriate preprocessing file type
    file_type = find_file_type(caps_config)

    def create_artifacts_image(data_idx: int) -> pd.DataFrame:
        participant_id = data_df.at[data_idx, "participant_id"]
        session_id = data_df.at[data_idx, "session_id"]
        cohort = data_df.at[data_idx, "cohort"]
        image_path = Path(
            clinicadl_file_reader(
                [participant_id],
                [session_id],
                caps_config.data.caps_dict[cohort],
                file_type.model_dump(),
            )[0][0]
        )
        from clinicadl.utils.iotools.read_utils import get_info_from_filename

        (
            subject_name,
            session_name,
            filename_pattern,
            file_suffix,
        ) = get_info_from_filename(image_path)

        artif_image_nii_dir = (
            generated_caps_directory
            / "subjects"
            / subject_name
            / session_name
            / caps_config.preprocessing.preprocessing.value
        )
        artif_image_nii_dir.mkdir(parents=True, exist_ok=True)

        artifacts_tio = []
        arti_ext = ""
        for artif in generate_config.artifacts_list:
            if artif == "motion":
                artifacts_tio.append(
                    tio.RandomMotion(
                        degrees=generate_config.rotation,
                        translation=generate_config.translation,
                        num_transforms=generate_config.num_transforms,
                    )
                )
                arti_ext += "mot-"
            elif artif == "noise":
                artifacts_tio.append(
                    tio.RandomNoise(
                        std=generate_config.noise_std,
                    )
                )
                arti_ext += "noi-"
            elif artif == "contrast":
                artifacts_tio.append(tio.RandomGamma(log_gamma=generate_config.gamma))
                arti_ext += "con-"

        artif_image_nii_filename = f"{subject_name}_{session_name}_{filename_pattern}_art-{arti_ext[:-1]}{file_suffix}"

        artifacts = tio.transforms.Compose(artifacts_tio)

        artif_image = artifacts(tio.ScalarImage(image_path))
        artif_image.save(artif_image_nii_dir / artif_image_nii_filename)

        # Append row to output tsv
        row = [subject_name, session_name, generate_config.artifacts_list]
        columns = ["participant_id", "session_id", "diagnosis"]
        row_df = pd.DataFrame([row], columns=columns)

        return row_df

    results_df = Parallel(n_jobs=caps_config.dataloader.n_proc)(
        delayed(create_artifacts_image)(data_idx) for data_idx in range(len(data_df))
    )
    output_df = pd.DataFrame()
    for result in results_df:
        output_df = pd.concat([result, output_df])

    output_df.to_csv(generated_caps_directory / "data.tsv", sep="\t", index=False)

    write_missing_mods(generated_caps_directory, output_df)

    logger.info(
        f"Images corrupted with artefacts were generated at {generated_caps_directory}"
    )


if __name__ == "__main__":
    cli()
