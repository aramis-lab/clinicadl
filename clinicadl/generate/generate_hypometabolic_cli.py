from logging import getLogger
from pathlib import Path

import click
import nibabel as nib
import pandas as pd
from joblib import Parallel, delayed
from nilearn.image import resample_to_img

from clinicadl.generate import generate_param
from clinicadl.generate.generate_config import (
    GenerateHypometabolicConfig,
    Preprocessing,
    SUVRReferenceRegions,
    Tracer,
)
from clinicadl.utils.caps_dataset.data import CapsDataset
from clinicadl.utils.clinica_utils import (
    RemoteFileStructure,
    clinicadl_file_reader,
    fetch_file,
)
from clinicadl.utils.exceptions import DownloadError
from clinicadl.utils.maps_manager.iotools import commandline_to_json
from clinicadl.utils.tsvtools_utils import extract_baseline

from .generate_utils import (
    find_file_type,
    load_and_check_tsv,
    mask_processing,
    write_missing_mods,
)

logger = getLogger("clinicadl.generate.hypometabolic")


@click.command(name="hypometabolic", no_args_is_help=True)
@generate_param.argument.caps_directory
@generate_param.argument.generated_caps_directory
@generate_param.option.n_proc
@generate_param.option.participants_tsv
@generate_param.option.n_subjects
@generate_param.option.use_uncropped_image
@generate_param.option_hypometabolic.sigma
@generate_param.option_hypometabolic.anomaly_degree
@generate_param.option_hypometabolic.pathology
def cli(caps_directory, generated_caps_directory, **kwargs):
    """Generation of trivial dataset with addition of synthetic brain atrophy.

    CAPS_DIRECTORY is the CAPS folder from where input brain images will be loaded.

    GENERATED_CAPS_DIRECTORY is a CAPS folder where the trivial dataset will be saved.
    """

    hypo_config = GenerateHypometabolicConfig(
        caps_directory=caps_directory,
        generated_caps_directory=generated_caps_directory,  # output_dir
        participants_list=kwargs["participants_tsv"],  # tsv_path
        preprocessing_cls=Preprocessing("pet-linear"),
        pathology_cls=kwargs["pathology"],
        **kwargs,
    )

    commandline_to_json(
        {
            "output_dir": hypo_config.generated_caps_directory,
            "caps_dir": hypo_config.caps_directory,
            "preprocessing": hypo_config.preprocessing,
            "n_subjects": hypo_config.n_subjects,
            "n_proc": hypo_config.n_proc,
            "pathology": hypo_config.pathology,
            "anomaly_degree": hypo_config.anomaly_degree,
        }
    )

    # Transform caps_directory in dict
    caps_dict = CapsDataset.create_caps_dict(
        hypo_config.caps_directory, multi_cohort=False
    )
    # Read DataFrame
    data_df = load_and_check_tsv(
        hypo_config.participants_list, caps_dict, hypo_config.generated_caps_directory
    )
    data_df = extract_baseline(data_df)

    if hypo_config.n_subjects > len(data_df):
        raise IndexError(
            f"The number of subjects {hypo_config.n_subjects} cannot be higher "
            f"than the number of subjects in the baseline dataset of size {len(data_df)}"
            f"Please add the '--n_subjects' option and re-run the command."
        )
    checksum_dir = {
        "ad": "2100d514a3fabab49fe30702700085a09cdad449bdf1aa04b8f804e238e4dfc2",
        "bvftd": "5a0ad28dff649c84761aa64f6e99da882141a56caa46675b8bf538a09fce4f81",
        "lvppa": "1099f5051c79d5b4fdae25226d97b0e92f958006f6545f498d4b600f3f8a422e",
        "nfvppa": "9512a4d4dc0003003c4c7526bf2d0ddbee65f1c79357f5819898453ef7271033",
        "pca": "ace36356b57f4db73e17c421a7cfd7ae056a1b258b8126534cf65d8d0be9527a",
        "svppa": "44f2e00bf2d2d09b532cb53e3ba61d6087b4114768cc8ae3330ea84c4b7e0e6a",
    }
    home = Path.home()
    cache_clinicadl = home / ".cache" / "clinicadl" / "ressources" / "masks_hypo"  # noqa (typo in resources)
    url_aramis = "https://aramislab.paris.inria.fr/files/data/masks/hypo/"
    FILE1 = RemoteFileStructure(
        filename=f"mask_hypo_{hypo_config.pathology}.nii",
        url=url_aramis,
        checksum=checksum_dir[hypo_config.pathology],
    )
    cache_clinicadl.mkdir(parents=True, exist_ok=True)
    if not (cache_clinicadl / f"mask_hypo_{hypo_config.pathology}.nii").is_file():
        logger.info(f"Downloading {hypo_config.pathology} masks...")
        try:
            mask_path = fetch_file(FILE1, cache_clinicadl)
        except Exception:
            DownloadError(
                """Unable to download masks, please download them
                manually at https://aramislab.paris.inria.fr/files/data/masks/
                and provide a valid path."""
            )

    else:
        mask_path = cache_clinicadl / f"mask_hypo_{hypo_config.pathology}.nii"

    mask_nii = nib.load(mask_path)

    # Find appropriate preprocessing file type
    file_type = find_file_type(
        hypo_config.preprocessing,
        hypo_config.use_uncropped_image,
        Tracer.FFDG,
        SUVRReferenceRegions.CEREBELLUMPONS2,
    )

    # Output tsv file
    columns = ["participant_id", "session_id", "pathology", "percentage"]
    output_df = pd.DataFrame(columns=columns)
    participants = [
        data_df.loc[i, "participant_id"] for i in range(hypo_config.n_subjects)
    ]
    sessions = [data_df.loc[i, "session_id"] for i in range(hypo_config.n_subjects)]
    cohort = caps_directory

    images_paths = clinicadl_file_reader(participants, sessions, cohort, file_type)[0]
    image_nii = nib.load(images_paths[0])

    mask_resample_nii = resample_to_img(mask_nii, image_nii, interpolation="nearest")
    mask = mask_resample_nii.get_fdata()

    mask = mask_processing(mask, hypo_config.anomaly_degree, hypo_config.sigma)

    # Create subjects dir
    (hypo_config.generated_caps_directory / "subjects").mkdir(
        parents=True, exist_ok=True
    )

    def generate_hypometabolic_image(
        subject_id: int, output_df: pd.DataFrame
    ) -> pd.DataFrame:
        image_path = Path(images_paths[subject_id])
        image_nii = nib.load(image_path)
        image = image_nii.get_fdata()
        if image_path.suffix == ".gz":
            input_filename = Path(image_path.stem).stem
        else:
            input_filename = image_path.stem
        input_filename = input_filename.strip("pet")
        hypo_image_nii_dir = (
            hypo_config.generated_caps_directory
            / "subjects"
            / participants[subject_id]
            / sessions[subject_id]
            / hypo_config.preprocessing
        )
        hypo_image_nii_filename = f"{input_filename}pat-{hypo_config.pathology}_deg-{int(hypo_config.anomaly_degree)}_pet.nii.gz"
        hypo_image_nii_dir.mkdir(parents=True, exist_ok=True)

        # Create atrophied image
        hypo_image = image * mask
        hypo_image_nii = nib.Nifti1Image(hypo_image, affine=image_nii.affine)
        hypo_image_nii.to_filename(hypo_image_nii_dir / hypo_image_nii_filename)

        # Append row to output tsv
        row = [
            participants[subject_id],
            sessions[subject_id],
            hypo_config.pathology,
            hypo_config.anomaly_degree,
        ]
        row_df = pd.DataFrame([row], columns=columns)
        output_df = pd.concat([output_df, row_df])
        return output_df

    results_list = Parallel(n_jobs=hypo_config.n_proc)(
        delayed(generate_hypometabolic_image)(subject_id, output_df)
        for subject_id in range(hypo_config.n_subjects)
    )

    output_df = pd.DataFrame()
    for result_df in results_list:
        output_df = pd.concat([result_df, output_df])

    output_df.to_csv(
        hypo_config.generated_caps_directory / "data.tsv", sep="\t", index=False
    )

    write_missing_mods(hypo_config.generated_caps_directory, output_df)

    logger.info(
        f"Hypometabolic dataset was generated, with {hypo_config.anomaly_degree} % of "
        f"dementia {hypo_config.pathology} at {hypo_config.generated_caps_directory}."
    )


if __name__ == "__main__":
    cli()
