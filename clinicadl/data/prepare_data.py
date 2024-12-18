from logging import getLogger
from pathlib import Path

import nibabel as nib
import pandas as pd
import torch
from joblib import Parallel, delayed
from pydantic import PositiveInt
from torch import save as save_tensor
from tqdm import tqdm

from .datasets import CapsDataset

logger = getLogger("clinicadl.prepare_data")


def prepare_data(
    caps_dataset: CapsDataset,
    n_proc: PositiveInt = 2,
):
    """
    Prepares tensor files from the neuroimaging data.

    This method processes the raw neuroimaging data (NIfTI format) into PyTorch tensors
    and stores them for faster data loading during training and evaluation.

    Parameters
    ----------
    caps_dataset : CapsDataset
        The CapsDataset to prepare for experiment.
    n_proc : PositiveInt, optional
        Number of processes to use for parallelization (default is 2).

    Notes
    -----
    - If the tensor file for a participant/session already exists, it will not be reprocessed.
    - This method saves tensor files and image statistics (mean, std, min, max) for each image.
    """

    def prepare_image(participant, session):
        image_path = caps_dataset.caps_reader.get_image_path(
            participant, session, caps_dataset.preprocessing
        )
        output_file_dir = caps_dataset.caps_reader.get_tensor_dir(
            participant, session, preprocessing=caps_dataset.preprocessing
        )

        output_file_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_file_dir / Path(image_path).name.replace(".nii.gz", ".pt")

        if output_file.is_file():
            logger.info(
                f"The file '{output_file}' already exists, the tensor has already been extracted."
            )
        else:
            logger.debug(f"Processing of {image_path}.")
            image_array = nib.loadsave.load(image_path).get_fdata(dtype="float32")  # type: ignore

            # get some important infos about the image
            info_df = pd.DataFrame(
                [
                    {
                        "mean": image_array.mean(),
                        "std": image_array.std(),
                        "max": image_array.max(),
                        "min": image_array.min(),
                    }
                ]
            )
            info_df.to_csv(output_file_dir / "image_info.tsv", sep="\t", index=False)

            # extract and save the image tensor
            image_tensor = torch.from_numpy(image_array).unsqueeze(0).float()
            save_tensor(image_tensor.clone(), output_file)
            logger.debug(f"Output tensor saved at {output_file}")

    Parallel(n_jobs=n_proc)(
        delayed(prepare_image)(participant, session)
        for participant, session in tqdm(
            caps_dataset._get_participant_session_couples(), desc="Preparing data"
        )
    )
