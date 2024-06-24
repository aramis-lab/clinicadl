from logging import getLogger
from pathlib import Path

import click
import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed

from clinicadl.caps_dataset.extraction.utils import write_preprocessing
from clinicadl.commandline import arguments
from clinicadl.commandline.modules_options import data, dataloader
from clinicadl.commandline.pipelines.generate.shepplogan import options as shepplogan
from clinicadl.generate.generate_config import GenerateSheppLoganConfig
from clinicadl.generate.generate_utils import (
    generate_shepplogan_phantom,
    write_missing_mods,
)
from clinicadl.utils.clinica_utils import FileType
from clinicadl.utils.iotools import check_and_clean, commandline_to_json

logger = getLogger("clinicadl.generate.shepplogan")


@click.command(name="shepplogan", no_args_is_help=True)
@arguments.generated_caps_directory
@data.n_subjects
@dataloader.n_proc
@shepplogan.extract_json
@shepplogan.image_size
@shepplogan.cn_subtypes_distribution
@shepplogan.ad_subtypes_distribution
@shepplogan.smoothing
def cli(generated_caps_directory, n_subjects, n_proc, **kwargs):
    """Random generation of 2D Shepp-Logan phantoms.
    Generate a dataset of 2D images at GENERATED_CAPS_DIRECTORY including
    3 subtypes based on Shepp-Logan phantom.
    """
    # caps_config = create_caps_dataset_config(extract=ExtractionMethod.IMAGE, preprocessing=Preprocessing.PET_LINEAR)(**kwargs)
    generate_config = GenerateSheppLoganConfig(**kwargs)

    labels_distribution = {
        "AD": generate_config.ad_subtypes_distribution,
        "CN": generate_config.cn_subtypes_distribution,
    }
    check_and_clean(generated_caps_directory / "subjects")
    commandline_to_json(
        {
            "output_dir": generated_caps_directory,
            "img_size": generate_config.image_size,
            "labels_distribution": labels_distribution,
            "samples": n_subjects,
            "smoothing": generate_config.smoothing,
        }
    )
    columns = ["participant_id", "session_id", "diagnosis", "subtype"]
    data_df = pd.DataFrame(columns=columns)

    for label_id, label in enumerate(labels_distribution.keys()):

        def create_shepplogan_image(
            subject_id: int, data_df: pd.DataFrame
        ) -> pd.DataFrame:
            # for j in range(samples):
            participant_id = f"sub-CLNC{label_id}{subject_id:04d}"
            session_id = "ses-M000"
            subtype = np.random.choice(
                np.arange(len(labels_distribution[label])), p=labels_distribution[label]
            )
            row_df = pd.DataFrame(
                [[participant_id, session_id, label, subtype]], columns=columns
            )
            data_df = pd.concat([data_df, row_df])

            # Image generation
            slice_path = (
                generated_caps_directory
                / "subjects"
                / participant_id
                / session_id
                / "deeplearning_prepare_data"
                / "slice_based"
                / "custom"
                / f"{participant_id}_{session_id}_space-SheppLogan_axis-axi_channel-single_slice-0_phantom.pt"
            )

            slice_dir = slice_path.parent
            slice_dir.mkdir(parents=True, exist_ok=True)
            slice_np = generate_shepplogan_phantom(
                generate_config.image_size,
                label=subtype,
                smoothing=generate_config.smoothing,
            )
            slice_tensor = torch.from_numpy(slice_np).float().unsqueeze(0)
            torch.save(slice_tensor, slice_path)

            image_path = (
                generated_caps_directory
                / "subjects"
                / participant_id
                / session_id
                / "shepplogan"
                / f"{participant_id}_{session_id}_space-SheppLogan_phantom.nii.gz"
            )
            image_dir = image_path.parent
            image_dir.mkdir(parents=True, exist_ok=True)
            with image_path.open("w") as f:
                f.write("0")
            return data_df

        results_df = Parallel(n_jobs=n_proc)(
            delayed(create_shepplogan_image)(subject_id, data_df)
            for subject_id in range(n_subjects)
        )

        data_df = pd.DataFrame()
        for result in results_df:
            data_df = pd.concat([result, data_df])

    # Save data
    data_df.to_csv(generated_caps_directory / "data.tsv", sep="\t", index=False)

    # Save preprocessing JSON file
    preprocessing_dict = {
        "preprocessing": "custom",
        "mode": "slice",
        "use_uncropped_image": False,
        "prepare_dl": True,
        "extract_json": generate_config.extract_json,
        "slice_direction": 2,
        "slice_mode": "single",
        "discarded_slices": 0,
        "num_slices": 1,
        "file_type": FileType(
            pattern="*_space-SheppLogan_phantom.nii.gz",
            description="Custom suffix",
            needed_pipeline="shepplogan",
        ).model_dump(),
    }
    write_preprocessing(preprocessing_dict, generated_caps_directory)
    write_missing_mods(generated_caps_directory, data_df)

    logger.info(f"Shepplogan dataset was generated at {generated_caps_directory}")


if __name__ == "__main__":
    cli()
