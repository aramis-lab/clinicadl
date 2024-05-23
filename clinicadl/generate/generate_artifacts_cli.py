from logging import getLogger
from pathlib import Path

import click
import pandas as pd
import torchio as tio
from joblib import Parallel, delayed

from clinicadl.config import arguments
from clinicadl.config.options import data, dataloader, generate, modality, preprocessing
from clinicadl.generate.generate_config import GenerateArtifactsConfig
from clinicadl.utils.caps_dataset.data import CapsDataset
from clinicadl.utils.clinica_utils import clinicadl_file_reader
from clinicadl.utils.maps_manager.iotools import commandline_to_json

from .generate_utils import (
    find_file_type,
    load_and_check_tsv,
    write_missing_mods,
)

logger = getLogger("clinicadl.generate.artifacts")


@click.command(name="artifacts", no_args_is_help=True)
@arguments.caps_directory
@arguments.generated_caps_directory
@dataloader.n_proc
@preprocessing.preprocessing
@preprocessing.use_uncropped_image
@data.participants_tsv
@modality.tracer
@modality.suvr_reference_region
@generate.artifacts.contrast
@generate.artifacts.motion
@generate.artifacts.noise_std
@generate.artifacts.noise
@generate.artifacts.num_transforms
@generate.artifacts.translation
@generate.artifacts.rotation
@generate.artifacts.gamma
def cli(**kwargs):
    """
    Addition of artifacts (noise, motion or contrast) to brain images

    """

    artif_config = GenerateArtifactsConfig(
        participants_list=kwargs["participants_tsv"],
        **kwargs,
    )

    multi_cohort = False  # hard coded ??????
    commandline_to_json(
        {
            "output_dir": artif_config.generated_caps_directory,
            "caps_dir": artif_config.caps_directory,
            "preprocessing": artif_config.preprocessing,
        }
    )

    # Transform caps_directory in dict
    caps_dict = CapsDataset.create_caps_dict(
        artif_config.caps_directory, multi_cohort=multi_cohort
    )
    # Read DataFrame
    data_df = load_and_check_tsv(
        artif_config.participants_list, caps_dict, artif_config.generated_caps_directory
    )
    # Create subjects dir
    (artif_config.generated_caps_directory / "subjects").mkdir(
        parents=True, exist_ok=True
    )

    # Output tsv file
    columns = ["participant_id", "session_id", "diagnosis"]
    output_df = pd.DataFrame(columns=columns)

    # Find appropriate preprocessing file type
    file_type = find_file_type(
        artif_config.preprocessing,
        artif_config.use_uncropped_image,
        artif_config.tracer,
        artif_config.suvr_reference_region,
    )
    artifacts_list = []
    if artif_config.motion:
        artifacts_list.append("motion")
    if artif_config.contrast:
        artifacts_list.append("contrast")
    if artif_config.noise:
        artifacts_list.append("noise")

    def create_artifacts_image(data_idx: int, output_df: pd.DataFrame) -> pd.DataFrame:
        participant_id = data_df.at[data_idx, "participant_id"]
        session_id = data_df.at[data_idx, "session_id"]
        cohort = data_df.at[data_idx, "cohort"]
        image_path = Path(
            clinicadl_file_reader(
                [participant_id], [session_id], caps_dict[cohort], file_type
            )[0][0]
        )
        input_filename = image_path.name
        filename_pattern = "_".join(input_filename.split("_")[2::])
        subject_name = input_filename.split("_")[:1][0]
        session_name = input_filename.split("_")[1:2][0]

        artif_image_nii_dir = (
            artif_config.generated_caps_directory
            / "subjects"
            / subject_name
            / session_name
            / artif_config.preprocessing.value
        )
        artif_image_nii_dir.mkdir(parents=True, exist_ok=True)

        artifacts_tio = []
        arti_ext = ""
        for artif in artifacts_list:
            if artif == "motion":
                artifacts_tio.append(
                    tio.RandomMotion(
                        degrees=artif_config.rotation,
                        translation=artif_config.translation,
                        num_transforms=artif_config.num_transforms,
                    )
                )
                arti_ext += "mot-"
            elif artif == "noise":
                artifacts_tio.append(
                    tio.RandomNoise(
                        std=artif_config.noise_std,
                    )
                )
                arti_ext += "noi-"
            elif artif == "contrast":
                artifacts_tio.append(tio.RandomGamma(log_gamma=artif_config.gamma))
                arti_ext += "con-"

        if filename_pattern.endswith(".nii.gz"):
            file_suffix = ".nii.gz"
            filename_pattern = Path(Path(filename_pattern).stem).stem
        elif filename_pattern.endswith(".nii"):
            file_suffix = ".nii"
            filename_pattern = Path(filename_pattern).stem

        artif_image_nii_filename = f"{subject_name}_{session_name}_{filename_pattern}_art-{arti_ext[:-1]}{file_suffix}"

        artifacts = tio.transforms.Compose(artifacts_tio)

        artif_image = artifacts(tio.ScalarImage(image_path))
        artif_image.save(artif_image_nii_dir / artif_image_nii_filename)

        # Append row to output tsv
        row = [subject_name, session_name, artifacts_list]
        row_df = pd.DataFrame([row], columns=columns)
        output_df = pd.concat([output_df, row_df])

        return output_df

    results_df = Parallel(n_jobs=artif_config.n_proc)(
        delayed(create_artifacts_image)(data_idx, output_df)
        for data_idx in range(len(data_df))
    )
    output_df = pd.DataFrame()
    for result in results_df:
        output_df = pd.concat([result, output_df])

    output_df.to_csv(
        artif_config.generated_caps_directory / "data.tsv", sep="\t", index=False
    )

    write_missing_mods(artif_config.generated_caps_directory, output_df)

    logger.info(
        f"Images corrupted with artefacts were generated at {artif_config.generated_caps_directory}"
    )


if __name__ == "__main__":
    cli()
