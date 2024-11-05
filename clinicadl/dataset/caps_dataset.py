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
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

from clinicadl.dataset.config.extraction import (
    ExtractionConfig,
    ExtractionImageConfig,
    ExtractionPatchConfig,
    ExtractionROIConfig,
    ExtractionSliceConfig,
)
from clinicadl.dataset.config.preprocessing import PreprocessingConfig
from clinicadl.dataset.utils import CapsDatasetOutput
from clinicadl.transforms.transforms import Transforms
from clinicadl.utils.enum import (
    STR,
    ExtractionMethod,
    Pattern,
    Preprocessing,
    SliceDirection,
    SliceMode,
    SubFolder,
    Suffix,
    Template,
)
from clinicadl.utils.exceptions import (
    ClinicaDLCAPSError,
    ClinicaDLTSVError,
)
from clinicadl.utils.iotools.clinica_utils import clinicadl_file_reader

logger = getLogger("clinicadl")


class CapsDataset(Dataset):
    """Abstract class for all derived CapsDatasets."""

    def __init__(
        self,
        caps_directory: Path,
        data_df: pd.DataFrame,
        preprocessing: PreprocessingConfig,
        transforms: Transforms,
        index: Optional[int] = None,
    ):
        self.caps_directory = caps_directory
        self.subjects_directory = caps_directory / STR.SUBJECTS.value
        self.preprocessing = preprocessing
        self.transforms = transforms
        self.extraction = transforms.extraction
        self.image_0 = self._get_full_image()
        self.df = data_df
        mandatory_col = {
            STR.PARTICIPANT_ID.value,
            STR.SESSION_ID.value,
            STR.COHORT.value,
        }

        if not mandatory_col.issubset(set(self.df.columns.values)):
            raise ClinicaDLTSVError(
                f"the data file is not in the correct format."
                f"Columns should include {mandatory_col}"
            )
        self.elem_index = index
        self.elem_per_image = self.extraction.num_elem_per_image(
            elem_index=self.elem_index, image=self.image_0
        )
        self.size = self[0].image.size()

    def __len__(self) -> int:
        return len(self.df) * self.elem_per_image

    def _get_image_path(self, participant: str, session: str, cohort: str) -> Path:
        """
        Gets the path to the tensor image (*.pt)

        Args:
            participant: ID of the participant.
            session: ID of the session.
            cohort: Name of the cohort.
        Returns:
            image_path: path to the tensor containing the whole image.
        """

        # Try to find .nii.gz file
        try:
            results = clinicadl_file_reader(
                [participant],
                [session],
                self.caps_directory,
                self.preprocessing.file_type.model_dump(),
            )
            logger.debug(f"clinicadl_file_reader output: {results}")
            filepath = Path(results[0][0])
            image_filename = filepath.name.replace(Suffix.PT.value, Suffix.NII_GZ.value)

            image_dir = (
                self.caps_directory
                / STR.SUBJECTS.value
                / participant
                / session
                / STR.DEEP_L_P_DATA.value
                / SubFolder.IMAGE.value
                / self.preprocessing.compute_folder()
            )
            image_path = image_dir / image_filename
        # Try to find .pt file
        except ClinicaDLCAPSError:
            self.preprocessing.file_type.pattern = (
                self.preprocessing.file_type.pattern.replace(
                    Suffix.NII_GZ.value, Suffix.PT.value
                )
            )
            results = clinicadl_file_reader(
                [participant],
                [session],
                self.caps_directory,
                self.preprocessing.file_type.model_dump(),
            )
            filepath = results[0]
            image_path = Path(filepath[0])

        return image_path

    def _get_meta_data(self, idx: int) -> Tuple[str, str, str, int]:
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
        image_idx = idx // self.elem_per_image
        participant = self.df.at[image_idx, STR.PARTICIPANT_ID.value]
        session = self.df.at[image_idx, STR.SESSION_ID.value]
        cohort = self.df.at[image_idx, STR.COHORT.value]

        if self.elem_index is None:
            elem_idx = idx % self.elem_per_image
        else:
            elem_idx = self.elem_index

        return participant, session, cohort, elem_idx

    def _get_full_image(self) -> torch.Tensor:
        """
        Allows to get the an example of the image mode corresponding to the dataset.
        Useful to compute the number of elements if mode != image.

        Returns:
            image tensor of the full image first image.
        """

        participant_id = self.df.at[0, STR.PARTICIPANT_ID.value]
        session_id = self.df.at[0, STR.SESSION_ID.value]
        cohort = self.df.at[0, STR.COHORT.value]

        try:
            image_path = self._get_image_path(participant_id, session_id, cohort)
            image = torch.load(image_path, weights_only=True)
        except IndexError:
            results = clinicadl_file_reader(
                [participant_id],
                [session_id],
                self.caps_directory,
                self.preprocessing.file_type.model_dump(),
            )
            image_nii = nib.loadsave.load((results[0]))  # type: ignore
            image_np = image_nii.get_fdata()  # type: ignore
            image = ToTensor()(image_np)

        return image

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
        participant, session, cohort, index = self._get_meta_data(idx)

        image_path = self._get_image_path(participant, session, cohort)
        image = torch.load(image_path, weights_only=True)

        (
            image_trf,
            object_trf,
            image_augmentation,
            object_augmentation,
        ) = self.transforms.get_transforms()

        image = image_trf(image)

        if image_augmentation and not self.eval_mode:
            image = image_augmentation(image)

        if not isinstance(self.extraction, ExtractionImageConfig):
            tensor = self.transforms.extraction.extract_tensor(
                image,
                index,
            )
            if object_trf:
                tensor = object_trf(tensor)

            if object_augmentation and not self.eval_mode:
                tensor = object_augmentation(tensor)

            out = tensor
            index = 0

        else:
            out = image

        sample = CapsDatasetOutput(
            image=out,
            # label=label,
            participant_id=participant,
            session_id=session,
            image_id=index,
            image_path=image_path,
            mode=self.extraction.extract_method,
        )

        return sample

    def num_elem_per_image(self) -> int:
        """Computes the number of elements per image based on the full image."""
        return self.extraction.num_elem_per_image(
            elem_index=self.elem_index, image=self.image_0
        )

    def eval(self):
        """Put the dataset on evaluation mode (data augmentation is not performed)."""
        self.eval_mode = True
        return self

    def train(self):
        """Put the dataset on training mode (data augmentation is performed)."""
        self.eval_mode = False
        return self
