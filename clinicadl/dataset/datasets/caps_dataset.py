# coding: utf8
# TODO: create a folder for generate/ prepare_data/ data to deal with capsDataset objects ?
import abc
from logging import getLogger
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from torch import save as save_tensor
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

from clinicadl.dataset.config.preprocessing import PreprocessingConfig
from clinicadl.dataset.readers.caps_reader import CapsReader
from clinicadl.dataset.transforms.extraction import (
    ROI,
    BaseExtraction,
    Image,
    Patch,
    Slice,
)
from clinicadl.dataset.transforms.transforms import Transforms
from clinicadl.dataset.utils import CapsDatasetOutput, check_df, tsv_to_df
from clinicadl.utils.enum import (
    ExtractionMethod,
    Pattern,
    Preprocessing,
    SliceDirection,
    SliceMode,
    Template,
)
from clinicadl.utils.exceptions import (
    ClinicaDLCAPSError,
    ClinicaDLConfigurationError,
    ClinicaDLTSVError,
)
from clinicadl.utils.iotools.clinica_utils import (
    clinicadl_file_reader,
    create_subs_sess_list,
)

logger = getLogger("clinicadl")

PARTICIPANT_ID = "participant_id"
SESSION_ID = "session_id"


class CapsDataset(Dataset):
    def __init__(
        self,
        caps_directory: Path,
        preprocessing: PreprocessingConfig,
        transforms: Transforms,
        data: Optional[Union[pd.DataFrame, Path]] = None,
    ):
        self.eval_mode = False
        self.caps_reader = CapsReader(caps_directory)
        self.preprocessing = preprocessing
        self.transforms = transforms
        self.extraction = transforms.extraction
        self.df = self._get_df_from_input(data)
        self.elem_per_image = self.extraction.num_elem_per_image(
            image=self._get_full_image()[0]
        )
        # self.size = self[0].elem.size()

    def _get_df_from_input(self, data: Optional[Union[pd.DataFrame, Path]] = None):
        """
        Gets the DataFrame from the input data.

        Args:
            data: Path to the TSV file or DataFrame.
        """
        if data is None:
            data = create_subs_sess_list(
                self.caps_reader.input_directory, self.caps_reader.input_directory
            )
            logger.info(f"Creating a subject session TSV file at {data}")

        if isinstance(data, Path):
            if not data.is_file():
                raise ClinicaDLTSVError(f"The data file does not exist: {data}")
            df = tsv_to_df(data)
        elif isinstance(data, pd.DataFrame):
            df = check_df(data)

        self.df = df
        if not self._check_preprocessing_config():
            raise ClinicaDLCAPSError(
                f"The DataFrame does not match the preprocessing configuration: {self.preprocessing.preprocessing.value}"
            )

        return df

    def _check_preprocessing_config(self) -> bool:
        pattern = self.preprocessing.file_type.pattern
        for participant, session in self._get_participants_sessions_couple():
            folder = self.caps_reader.get_session_path(
                participant=participant, session=session
            )
            if not list(folder.glob(pattern)):
                raise ClinicaDLConfigurationError(
                    f"Could not find preprocessing {self.preprocessing.preprocessing.value} for participant {participant} and session {session} with pattern: {pattern}"
                )
        return True

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.df) * self.elem_per_image

    def _get_meta_data(self, idx: int) -> Tuple[str, str, int, int]:
        """
        Gets all meta data necessary to compute the path with _get_image_path

        Args:
            idx (int): row number of the meta-data contained in self.df
        Returns:
            participant (str): ID of the participant.
            session (str): ID of the session.
            cohort (str): Name of the cohort.
            elem_index (int): Index of the part of the image.
            label (str or float or int): value of the label to be used in criterion.
        """

        if idx >= self.__len__():
            raise IndexError(
                f"Index out of range, there are only {self.__len__()} elements in your dataset."
            )

        img_idx = idx // self.elem_per_image
        elem_idx = idx % self.elem_per_image

        participant = self._get_participant(img_idx)
        session = self._get_session(img_idx)

        return participant, session, img_idx, elem_idx

    def _get_participant(self, idx):
        return self.df.at[idx, PARTICIPANT_ID]

    def _get_session(self, idx):
        return self.df.at[idx, SESSION_ID]

    def _get_participants_sessions_couple(self):
        return list(zip(self.df[PARTICIPANT_ID], self.df[SESSION_ID]))

    def _get_full_image(
        self, idx: int = 0, weights_only: bool = True
    ) -> tuple[torch.Tensor, Path]:
        """
        Allows to get the an example of the image mode corresponding to the dataset.
        Useful to compute the number of elements if mode != image.

        Returns:
            image tensor of the full first image.
        """

        participant_id = self._get_participant(idx)
        session_id = self._get_session(idx)

        image_path = self.caps_reader.get_tensor_path(
            participant_id, session_id, self.preprocessing
        )
        if image_path.is_file():
            image = torch.load(image_path, weights_only=weights_only)
        else:
            image_path = self.caps_reader.get_image_path(
                participant_id, session_id, self.preprocessing
            )
            image_nii = nib.loadsave.load(image_path)  # type: ignore
            image_np = image_nii.get_fdata()  # type: ignore
            image = (
                torch.from_numpy(image_np).unsqueeze(0).float()
            )  # ToTensor()(image_np) ???

        return image, image_path

    def __getitem__(self, idx: int) -> CapsDatasetOutput:
        """
        Gets the sample containing all the information needed for training and testing tasks.

        Args:
            idx: row number of the meta-data contained in self.df
        Returns:
            dictionary with following items:
                - "image" (torch.Tensor): the input given to the model,
                - "label" (int or float): the label used in criterion,
                - PARTICIPANT_ID (str): ID of the participant,
                - SESSION_ID (str): ID of the session,
                - f"{self.mode}_id" (int): number of the element,
                - "image_path": path to the image loaded in CAPS.

        """

        participant, session, img_index, elem_index = self._get_meta_data(idx)
        image, image_path = self._get_full_image(img_index, True)

        (
            image_trf,
            object_trf,
            image_augmentation,
            object_augmentation,
        ) = self.transforms.get_transforms()

        image = image_trf(image)

        if image_augmentation and not self.eval_mode:
            image = image_augmentation(image)

        if not isinstance(self.extraction, Image):
            tensor = self.transforms.extraction.extract_tensor(
                image,
                elem_index,
            )
            if object_trf:
                tensor = object_trf(tensor)

            if object_augmentation and not self.eval_mode:
                tensor = object_augmentation(tensor)

            out = tensor

        else:
            out = image

        sample = CapsDatasetOutput(
            elem=out,
            # label=label,
            participant_id=participant,
            session_id=session,
            img_idx=img_index,
            elem_idx=elem_index,
            image_path=image_path,
            mode=self.extraction.extract_method,
        )

        return sample

    def eval(self):
        """Put the dataset on evaluation mode (data augmentation is not performed)."""
        self.eval_mode = True
        return self

    def train(self):
        """Put the dataset on training mode (data augmentation is performed)."""
        self.eval_mode = False
        return self

    def prepare_data(
        self,
        n_proc: int = 2,
        use_uncropped_images: bool = False,
    ):
        """TO COMPLETE"""

        def prepare_image(participant, session):
            image_path = self.caps_reader.get_image_path(
                participant, session, self.preprocessing
            )
            output_file_dir = self.caps_reader.get_tensor_dir(
                participant, session, preprocessing=self.preprocessing
            )

            output_file_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_file_dir / Path(image_path).name.replace(
                ".nii.gz", ".pt"
            )

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
                info_df.to_csv(
                    output_file_dir / "image_info.tsv", sep="\t", index=False
                )

                # extract and save the image tensor
                image_tensor = torch.from_numpy(image_array).unsqueeze(0).float()
                save_tensor(image_tensor.clone(), output_file)
                logger.debug(f"Output tensor saved at {output_file}")

        Parallel(n_jobs=n_proc)(
            delayed(prepare_image)(participant, session)
            for participant, session in self._get_participants_sessions_couple()
        )
