from logging import getLogger

import click
import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed

from clinicadl.generate import generate_param
from clinicadl.generate.generate_config import GenerateSheppLoganConfig
from clinicadl.utils.maps_manager.iotools import check_and_clean, commandline_to_json
from clinicadl.utils.preprocessing.preprocessing import write_preprocessing

from .generate_utils import (
    generate_shepplogan_phantom,
    write_missing_mods,
)

logger = getLogger("clinicadl.generate.shepplogan")


@click.command(name="shepplogan", no_args_is_help=True)
@generate_param.argument.generated_caps_directory
@generate_param.option.n_subjects
@generate_param.option.n_proc
@generate_param.option_shepplogan.extract_json
@generate_param.option_shepplogan.image_size
@generate_param.option_shepplogan.cn_subtypes_distribution
@generate_param.option_shepplogan.ad_subtypes_distribution
@generate_param.option_shepplogan.smoothing
def cli(generated_caps_directory, **kwargs):
    """Random generation of 2D Shepp-Logan phantoms.

    Generate a dataset of 2D images at GENERATED_CAPS_DIRECTORY including
    3 subtypes based on Shepp-Logan phantom.
    """

    shepplogan_config = GenerateSheppLoganConfig(
        generated_caps_directory=generated_caps_directory, **kwargs
    )

    labels_distribution = {
        "AD": shepplogan_config.ad_subtypes_distribution,
        "CN": shepplogan_config.cn_subtypes_distribution,
    }
    check_and_clean(shepplogan_config.generated_caps_directory / "subjects")
    commandline_to_json(
        {
            "output_dir": shepplogan_config.generated_caps_directory,
            "img_size": shepplogan_config.image_size,
            "labels_distribution": labels_distribution,
            "samples": shepplogan_config.n_subjects,
            "smoothing": shepplogan_config.smoothing,
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
                shepplogan_config.generated_caps_directory
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
                shepplogan_config.image_size,
                label=subtype,
                smoothing=shepplogan_config.smoothing,
            )
            slice_tensor = torch.from_numpy(slice_np).float().unsqueeze(0)
            torch.save(slice_tensor, slice_path)

            image_path = (
                shepplogan_config.generated_caps_directory
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

        results_df = Parallel(n_jobs=shepplogan_config.n_proc)(
            delayed(create_shepplogan_image)(subject_id, data_df)
            for subject_id in range(shepplogan_config.n_subjects)
        )

        data_df = pd.DataFrame()
        for result in results_df:
            data_df = pd.concat([result, data_df])

    # Save data
    data_df.to_csv(
        shepplogan_config.generated_caps_directory / "data.tsv", sep="\t", index=False
    )

    # Save preprocessing JSON file
    preprocessing_dict = {
        "preprocessing": "custom",
        "mode": "slice",
        "use_uncropped_image": False,
        "prepare_dl": True,
        "extract_json": shepplogan_config.extract_json,
        "slice_direction": 2,
        "slice_mode": "single",
        "discarded_slices": 0,
        "num_slices": 1,
        "file_type": {
            "pattern": f"*_space-SheppLogan_phantom.nii.gz",
            "description": "Custom suffix",
            "needed_pipeline": "shepplogan",
        },
    }
    write_preprocessing(preprocessing_dict, shepplogan_config.generated_caps_directory)
    write_missing_mods(shepplogan_config.generated_caps_directory, data_df)

    logger.info(
        f"Shepplogan dataset was generated at {shepplogan_config.generated_caps_directory}"
    )


if __name__ == "__main__":
    cli()
