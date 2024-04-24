import tarfile
from logging import getLogger
from pathlib import Path

import click
import nibabel as nib
import pandas as pd
from joblib import Parallel, delayed

from clinicadl.generate import generate_param
from clinicadl.generate.generate_config import GenerateTrivialConfig
from clinicadl.utils.caps_dataset.data import CapsDataset
from clinicadl.utils.clinica_utils import (
    RemoteFileStructure,
    clinicadl_file_reader,
    fetch_file,
)
from clinicadl.utils.maps_manager.iotools import commandline_to_json
from clinicadl.utils.tsvtools_utils import extract_baseline

from .generate_utils import (
    find_file_type,
    im_loss_roi_gaussian_distribution,
    load_and_check_tsv,
    write_missing_mods,
)

logger = getLogger("clinicadl.generate.trivial")


@click.command(name="trivial", no_args_is_help=True)
@generate_param.argument.caps_directory
@generate_param.argument.generated_caps_directory
@generate_param.option.preprocessing
@generate_param.option.participants_tsv
@generate_param.option.n_subjects
@generate_param.option.n_proc
@generate_param.option.use_uncropped_image
@generate_param.option.tracer
@generate_param.option.suvr_reference_region
@generate_param.option_trivial.atrophy_percent
@generate_param.option_trivial.mask_path
def cli(caps_directory, generated_caps_directory, **kwargs):
    trivial_config = GenerateTrivialConfig(
        caps_directory=caps_directory,
        generated_caps_directory=generated_caps_directory,
        suvr_reference_region_cls=kwargs["suvr_reference_region"],
        tracer_cls=kwargs["tracer"],
        participants_list=kwargs["participants_tsv"],
        preprocessing_cls=kwargs["preprocessing"],
    )

    from clinicadl.utils.exceptions import DownloadError

    commandline_to_json(
        {
            "output_dir": trivial_config.generated_caps_directory,
            "caps_dir": caps_directory,
            "preprocessing": trivial_config.preprocessing,
            "n_subjects": trivial_config.n_subjects,
            "n_proc": trivial_config.n_proc,
            "atrophy_percent": trivial_config.atrophy_percent,
        }
    )

    print(trivial_config.preprocessing)
    multi_cohort = False  # ??? hard coded

    # Transform caps_directory in dict
    caps_dict = CapsDataset.create_caps_dict(caps_directory, multi_cohort=multi_cohort)
    # Read DataFrame
    data_df = load_and_check_tsv(
        trivial_config.participants_list,
        caps_dict,
        trivial_config.generated_caps_directory,
    )
    data_df = extract_baseline(data_df)

    if trivial_config.n_subjects > len(data_df):
        raise IndexError(
            f"The number of subjects {trivial_config.n_subjects} cannot be higher "
            f"than the number of subjects in the baseline dataset of size {len(data_df)}"
        )

    if not trivial_config.mask_path.is_file():
        cache_clinicadl = Path.home() / ".cache" / "clinicadl" / "ressources" / "masks"  # noqa (typo in resources)
        url_aramis = "https://aramislab.paris.inria.fr/files/data/masks/"
        FILE1 = RemoteFileStructure(
            filename="AAL2.tar.gz",
            url=url_aramis,
            checksum="89427970921674792481bffd2de095c8fbf49509d615e7e09e4bc6f0e0564471",
        )
        cache_clinicadl.mkdir(parents=True, exist_ok=True)

        if not (cache_clinicadl / "AAL2").is_dir():
            print("Downloading AAL2 masks...")
            try:
                mask_path_tar = fetch_file(FILE1, cache_clinicadl)
                tar_file = tarfile.open(mask_path_tar)
                print(f"File: {mask_path_tar}")
                try:
                    tar_file.extractall(cache_clinicadl)
                    tar_file.close()
                    mask_path = cache_clinicadl / "AAL2"
                except RuntimeError:
                    print("Unable to extract downloaded files.")
            except IOError as err:
                print("Unable to download required templates:", err)
                raise DownloadError(
                    """Unable to download masks, please download them
                    manually at https://aramislab.paris.inria.fr/files/data/masks/
                    and provide a valid path."""
                )
        else:
            mask_path = cache_clinicadl / "AAL2"

    # Create subjects dir
    (trivial_config.generated_caps_directory / "subjects").mkdir(
        parents=True, exist_ok=True
    )

    # Output tsv file
    columns = ["participant_id", "session_id", "diagnosis", "age_bl", "sex"]
    output_df = pd.DataFrame(columns=columns)
    diagnosis_list = ["AD", "CN"]

    # Find appropriate preprocessing file type
    file_type = find_file_type(
        trivial_config.preprocessing,
        trivial_config.use_uncropped_image,
        trivial_config.tracer,
        trivial_config.suvr_reference_region,
    )

    def create_trivial_image(subject_id: int, output_df: pd.DataFrame) -> pd.DataFrame:
        data_idx = subject_id // 2
        label = subject_id % 2

        participant_id = data_df.loc[data_idx, "participant_id"]
        session_id = data_df.loc[data_idx, "session_id"]
        cohort = data_df.loc[data_idx, "cohort"]
        image_path = Path(
            clinicadl_file_reader(
                [participant_id], [session_id], caps_dict[cohort], file_type
            )[0][0]
        )
        image_nii = nib.load(image_path)
        image = image_nii.get_fdata()

        input_filename = image_path.name
        filename_pattern = "_".join(input_filename.split("_")[2::])

        trivial_image_nii_dir = (
            trivial_config.generated_caps_directory
            / "subjects"
            / f"sub-TRIV{subject_id}"
            / session_id
            / trivial_config.preprocessing
        )

        trivial_image_nii_filename = (
            f"sub-TRIV{subject_id}_{session_id}_{filename_pattern}"
        )

        trivial_image_nii_dir.mkdir(parents=True, exist_ok=True)

        path_to_mask = mask_path / f"mask-{label + 1}.nii"
        if path_to_mask.is_file():
            atlas_to_mask = nib.load(path_to_mask).get_fdata()
        else:
            raise ValueError("masks need to be named mask-1.nii and mask-2.nii")

        # Create atrophied image
        trivial_image = im_loss_roi_gaussian_distribution(
            image, atlas_to_mask, trivial_config.atrophy_percent
        )
        trivial_image_nii = nib.Nifti1Image(trivial_image, affine=image_nii.affine)
        trivial_image_nii.to_filename(
            trivial_image_nii_dir / trivial_image_nii_filename
        )

        # Append row to output tsv
        row = [f"sub-TRIV{subject_id}", session_id, diagnosis_list[label], 60, "F"]
        row_df = pd.DataFrame([row], columns=columns)
        output_df = pd.concat([output_df, row_df])

        return output_df

    results_df = Parallel(n_jobs=trivial_config.n_proc)(
        delayed(create_trivial_image)(subject_id, output_df)
        for subject_id in range(2 * trivial_config.n_subjects)
    )
    output_df = pd.DataFrame()
    for result in results_df:
        output_df = pd.concat([result, output_df])

    output_df.to_csv(
        trivial_config.generated_caps_directory / "data.tsv", sep="\t", index=False
    )
    write_missing_mods(trivial_config.generated_caps_directory, output_df)
    logger.info(
        f"Trivial dataset was generated at {trivial_config.generated_caps_directory}"
    )


if __name__ == "__main__":
    cli()
