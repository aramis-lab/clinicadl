from logging import getLogger
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import click
import pandas as pd
import torchio as tio
from clinica.utils.inputs import clinica_file_reader
from joblib import Parallel, delayed

from clinicadl.generate.generate_utils import (
    find_file_type,
    load_and_check_tsv,
    write_missing_mods,
)
from clinicadl.utils import cli_param
from clinicadl.utils.caps_dataset.data import CapsDataset
from clinicadl.utils.maps_manager.iotools import commandline_to_json

logger = getLogger("clinicadl.generate")

# def generate_noise_dataset(
#     caps_directory: Path,
#     output_dir: Path,
#     n_proc: int,
#     tsv_path: Optional[str] = None,
#     preprocessing: str = "t1-linear",
#     multi_cohort: bool = False,
#     uncropped_image: bool = False,
#     tracer: str = "fdg",
#     suvr_reference_region: str = "pons",
#     noise_std: List = [5, 15],
# ):
#     """
#     Generates a dataset, based on the images of the CAPS directory, where
#     all the images are corrupted with noise artefacts using the gaussian noise simulation of torchio.
#     Args:
#         caps_directory: Path
#             Path to the CAPS directory.
#         output_dir: Path
#             Folder containing the synthetic dataset in CAPS format.
#         n_proc: int
#             Number of cores used during the task.
#         tsv_path: Path
#             Path to tsv file of list of subjects/sessions.
#         preprocessing: str
#             Preprocessing performed. Must be in ['linear', 'extensive'].
#         multi_cohort: bool
#             If True caps_directory is the path to a TSV file linking cohort names and paths.
#         uncropped_image: bool
#             If True the uncropped image of `t1-linear` or `pet-linear` will be used.
#         tracer: str
#             Name of the tracer when using `pet-linear` preprocessing.
#         suvr_reference_region: str
#             Name of the reference region when using `pet-linear` preprocessing.
#     Returns:
#         Folder structure where images are stored in CAPS format.
#     """

#     commandline_to_json(
#         {
#             "output_dir": output_dir,
#             "caps_dir": caps_directory,
#             "preprocessing": preprocessing,
#         }
#     )

#     # Transform caps_directory in dict
#     caps_dict = CapsDataset.create_caps_dict(caps_directory, multi_cohort=multi_cohort)
#     # Read DataFrame
#     data_df = load_and_check_tsv(tsv_path, caps_dict, output_dir)
#     # Create subjects dir
#     (output_dir / "subjects").mkdir(parents=True, exist_ok=True)

#     # Output tsv file
#     columns = ["participant_id", "session_id", "diagnosis"]
#     output_df = pd.DataFrame(columns=columns)

#     # Find appropriate preprocessing file type
#     file_type = find_file_type(
#         preprocessing, uncropped_image, tracer, suvr_reference_region
#     )

#     def create_noise_image(data_idx, output_df):
#         participant_id = data_df.loc[data_idx, "participant_id"]
#         session_id = data_df.loc[data_idx, "session_id"]
#         cohort = data_df.loc[data_idx, "cohort"]

#         image_path = Path(
#             clinica_file_reader(
#                 [participant_id], [session_id], caps_dict[cohort], file_type
#             )[0][0]
#         )
#         input_filename = image_path.name
#         filename_pattern = "_".join(input_filename.split("_")[2::])

#         subject_id = participant_id.split("-")[1]

#         noise_image_nii_dir = (
#             output_dir
#             / "subjects"
#             / f"sub-NOIS{subject_id}"
#             / session_id
#             / preprocessing
#         )
#         noise_image_nii_filename = (
#             f"sub-NOIS{subject_id}_{session_id}_{filename_pattern}"
#         )

#         noise_image_nii_dir.mkdir(parents=True, exist_ok=True)

#         noise = tio.RandomNoise(std=(noise_std[0], noise_std[1]))

#         noise_image = noise(tio.ScalarImage(image_path))
#         noise_image.save(noise_image_nii_dir / noise_image_nii_filename)

#         # Append row to output tsv
#         row = [f"sub-NOIS{subject_id}", session_id, "noise"]
#         row_df = pd.DataFrame([row], columns=columns)
#         output_df = pd.concat([output_df, row_df])

#         return output_df

#     results_df = Parallel(n_jobs=n_proc)(
#         delayed(create_noise_image)(data_idx, output_df)
#         for data_idx in range(len(data_df))
#     )
#     output_df = pd.DataFrame()
#     for result in results_df:
#         output_df = pd.concat([result, output_df])

#     output_df.to_csv(output_dir / "data.tsv", sep="\t", index=False)

#     write_missing_mods(output_dir, output_df)

#     logger.info(f"Images corrupted with noise artefacts were generated at {output_dir}")


@click.command(name="artifacts", no_args_is_help=True)
@cli_param.argument.caps_directory
@cli_param.argument.generated_caps
@cli_param.option.n_proc
@cli_param.option.preprocessing
@cli_param.option.participant_list
@cli_param.option.use_uncropped_image
@cli_param.option.tracer
@cli_param.option.suvr_reference_region
@click.option(
    "--noise_std",
    type=float,
    multiple=2,
    default=[5, 15],
    help="Range for noise standard deviation",
)
def cli(
    caps_directory,
    generated_caps_directory,
    preprocessing,
    participants_tsv,
    use_uncropped_image,
    tracer,
    suvr_reference_region,
    noise_std,
    n_proc,
):
    """Generation of trivial dataset with addition of synthetic noise artifacts.
    CAPS_DIRECTORY is the CAPS folder from where input brain images will be loaded.
    GENERATED_CAPS_DIRECTORY is a CAPS folder where the trivial dataset will be saved.
    """
    from .generate import generate_noise_dataset

    generate_noise_dataset(
        caps_directory=caps_directory,
        tsv_path=participants_tsv,
        preprocessing=preprocessing,
        output_dir=generated_caps_directory,
        uncropped_image=use_uncropped_image,
        tracer=tracer,
        suvr_reference_region=suvr_reference_region,
        noise_std=noise_std,
        n_proc=n_proc,
    )


if __name__ == "__main__":
    cli()
