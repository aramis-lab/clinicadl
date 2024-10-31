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

from clinicadl.dataset.config.extraction import (
    ExtractionConfig,
    ExtractionImageConfig,
    ExtractionPatchConfig,
    ExtractionROIConfig,
    ExtractionSliceConfig,
)
from clinicadl.dataset.config.preprocessing import PreprocessingConfig
from clinicadl.dataset.utils import CapsDatasetOutput
from clinicadl.transforms.config import TransformsConfig
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
        extraction: ExtractionConfig,
        preprocessing: PreprocessingConfig,
        transforms: TransformsConfig,
    ):
        self.caps_directory = caps_directory
        self.subjects_directory = caps_directory / STR.SUBJECTS.value
        self.extraction = extraction
        self.preprocessing = preprocessing
        self.transforms = transforms

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
        self.elem_per_image = self.num_elem_per_image()
        self.size = self[0].image.size()

    @property
    @abc.abstractmethod
    def elem_index(self):
        pass

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

    def _get_meta_data(self, idx: int) -> Tuple[str, str, str, Union[float, int, None]]:
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
            image_nii = nib.loadsave.load(results[0])
            image_np = image_nii.get_fdata()
            image = ToTensor()(image_np)

        return image

    @abc.abstractmethod
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
        pass

    @abc.abstractmethod
    def num_elem_per_image(self) -> int:
        """Computes the number of elements per image based on the full image."""
        pass

    # def eval(self):
    #     """Put the dataset on evaluation mode (data augmentation is not performed)."""
    #     self.eval_mode = True
    #     return self

    # def train(self):
    #     """Put the dataset on training mode (data augmentation is performed)."""
    #     self.eval_mode = False
    #     return self


class CapsDatasetImage(CapsDataset):
    """Dataset of MRI organized in a CAPS folder."""

    def __init__(
        self,
        caps_directory: Path,
        data_df: pd.DataFrame,
        extraction: ExtractionImageConfig,
        preprocessing: PreprocessingConfig,
        transforms: TransformsConfig,
    ):
        """
        Args:
            caps_directory: Directory of all the images.
            data_file: Path to the tsv file or DataFrame containing the subject/session list.
            preprocessing_dict: preprocessing dict contained in the JSON file of prepare_data.
            train_transformations: Optional transform to be applied only on training mode.
            label_presence: If True the diagnosis will be extracted from the given DataFrame.
            label: Name of the column in data_df containing the label.
            label_code: label code that links the output node number to label value.
            all_transformations: Optional transform to be applied during training and evaluation.
            multi_cohort: If True caps_directory is the path to a TSV file linking cohort names and paths.

        """
        super().__init__(
            caps_directory=caps_directory,
            data_df=data_df,
            extraction=extraction,
            preprocessing=preprocessing,
            transforms=transforms,
        )

    @property
    def elem_index(self):
        return None

    def __getitem__(self, idx):
        participant, session, cohort, _ = self._get_meta_data(idx)

        image_path = self._get_image_path(participant, session, cohort)
        image = torch.load(image_path, weights_only=True)

        train_trf, trf = self.transforms.get_transforms()

        image = trf(image)
        if self.transforms.train_transformations and not self.eval_mode:  # train_mode
            image = train_trf(image)

        sample = CapsDatasetOutput(
            image=image,
            label=label,
            participant_id=participant,
            session_id=session,
            image_id=0,
            image_path=image_path,
            mode=ExtractionMethod.IMAGE,
        )

        return sample

    def num_elem_per_image(self):
        return 1


class CapsDatasetPatch(CapsDataset):
    def __init__(
        self,
        caps_directory: Path,
        data_df: pd.DataFrame,
        extraction: ExtractionPatchConfig,
        preprocessing: PreprocessingConfig,
        transforms: TransformsConfig,
        patch_index: Optional[int] = None,
    ):
        """
        caps_directory: Directory of all the images.
        data_file: Path to the tsv file or DataFrame containing the subject/session list.
        preprocessing_dict: preprocessing dict contained in the JSON file of prepare_data.
        train_transformations: Optional transform to be applied only on training mode.
        """
        # self.patch_index = patch_index
        self.extraction = extraction
        super().__init__(
            caps_directory=caps_directory,
            data_df=data_df,
            extraction=extraction,
            preprocessing=preprocessing,
            transforms=transforms,
        )

    # @property
    # def elem_index(self):
    #     return self.patch_index

    def __getitem__(self, idx):
        participant, session, cohort, patch_idx = self._get_meta_data(idx)
        image_path = self._get_image_path(participant, session, cohort)

        if self.extraction.save_features:
            patch_dir = image_path.parent.as_posix().replace(
                SubFolder.IMAGE.value, SubFolder.PATCH.value
            )
            patch_filename = self.extraction.extract_patch_path(
                image_path,
                patch_idx,
            )
            patch_tensor = torch.load(
                Path(patch_dir).resolve() / patch_filename, weights_only=True
            )

        else:
            image = torch.load(image_path, weights_only=True)
            patch_tensor = self.extraction.extract_patch_tensor(
                image,
                patch_idx,
            )

        train_trf, trf = self.transforms.get_transforms()
        patch_tensor = trf(patch_tensor)

        if self.transforms.train_transformations and not self.eval_mode:
            patch_tensor = train_trf(patch_tensor)

        sample = CapsDatasetOutput(
            image=patch_tensor,
            label=label,
            participant_id=participant,
            session_id=session,
            image_id=patch_idx,
            mode=ExtractionMethod.PATCH,
        )

        return sample


class CapsDatasetRoi(CapsDataset):
    def __init__(
        self,
        caps_directory: Path,
        data_df: pd.DataFrame,
        extraction: ExtractionROIConfig,
        preprocessing: PreprocessingConfig,
        transforms: TransformsConfig,
        roi_index: Optional[int] = None,
    ):
        """
        Args:
            caps_directory: Directory of all the images.
            data_file: Path to the tsv file or DataFrame containing the subject/session list.
            preprocessing_dict: preprocessing dict contained in the JSON file of prepare_data.
            roi_index: If a value is given the same region will be extracted for each image.
                else the dataset will load all the regions possible for one image.
            train_transformations: Optional transform to be applied only on training mode.
            label_presence: If True the diagnosis will be extracted from the given DataFrame.
            label: Name of the column in data_df containing the label.
            label_code: label code that links the output node number to label value.
            all_transformations: Optional transform to be applied during training and evaluation.
            multi_cohort: If True caps_directory is the path to a TSV file linking cohort names and paths.

        """
        self.roi_index = roi_index
        self.extraction = extraction
        super().__init__(
            caps_directory=caps_directory,
            data_df=data_df,
            extraction=extraction,
            preprocessing=preprocessing,
            transforms=transforms,
        )

        self.mask_paths, self.mask_arrays = self._get_mask_paths_and_tensors()

    @property
    def elem_index(self):
        return self.roi_index

    def __getitem__(self, idx):
        participant, session, cohort, roi_idx = self._get_meta_data(idx)
        image_path = self._get_image_path(participant, session, cohort)

        if self.extraction.roi_list is None:
            raise NotImplementedError(
                "Default regions are not available anymore in ClinicaDL. "
                "Please define appropriate masks and give a roi_list."
            )

        if self.extraction.save_features:
            mask_path = self.mask_paths[roi_idx]
            roi_dir = image_path.parent.as_posix().replace(
                SubFolder.IMAGE.value, SubFolder.ROI.value
            )
            roi_filename = self.extraction.extract_roi_path(image_path, mask_path)
            roi_tensor = torch.load(Path(roi_dir) / roi_filename, weights_only=True)

        else:
            image = torch.load(image_path, weights_only=True)
            mask_array = self.mask_arrays[roi_idx]
            roi_tensor = self.extraction.extract_roi_tensor(image, mask_array)

        train_trf, trf = self.transforms.get_transforms()

        roi_tensor = trf(roi_tensor)

        if self.transforms.train_transformations and not self.eval_mode:
            roi_tensor = train_trf(roi_tensor)

        sample = CapsDatasetOutput(
            image=roi_tensor,
            label=label,
            participant_id=participant,
            session_id=session,
            image_id=roi_idx,
            mode=ExtractionMethod.ROI,
        )

        return sample

    def num_elem_per_image(self):
        if self.elem_index is not None:
            return 1
        if self.extraction.roi_list is None:
            return 2
        else:
            return len(self.extraction.roi_list)

    def _get_mask_paths_and_tensors(
        self,
    ) -> Tuple[List[str], List]:
        """Loads the masks necessary to regions extraction"""

        mask_location = (
            self.caps_directory / "masks" / f"tpl-{self.extraction.roi_template}"
        )

        mask_paths, mask_arrays = list(), list()
        for roi in self.extraction.roi_list:
            logger.info(f"Find mask for roi {roi}.")
            mask_path, desc = self.extraction.find_mask_path(mask_location, roi)
            if mask_path is None:
                raise FileNotFoundError(desc)
            mask_nii = nib.loadsave.load(mask_path)
            mask_paths.append(Path(mask_path))
            mask_arrays.append(mask_nii.get_fdata())

        return mask_paths, mask_arrays


class CapsDatasetSlice(CapsDataset):
    def __init__(
        self,
        caps_directory: Path,
        data_df: pd.DataFrame,
        extraction: ExtractionSliceConfig,
        preprocessing: PreprocessingConfig,
        transforms: TransformsConfig,
        slice_index: Optional[int] = None,
    ):
        """
        Args:
            caps_directory: Directory of all the images.
            data_file: Path to the tsv file or DataFrame containing the subject/session list.
            preprocessing_dict: preprocessing dict contained in the JSON file of prepare_data.
            slice_index: If a value is given the same slice will be extracted for each image.
                else the dataset will load all the slices possible for one image.
            train_transformations: Optional transform to be applied only on training mode.
            label_presence: If True the diagnosis will be extracted from the given DataFrame.
            label: Name of the column in data_df containing the label.
            label_code: label code that links the output node number to label value.
            all_transformations: Optional transform to be applied during training and evaluation.
            multi_cohort: If True caps_directory is the path to a TSV file linking cohort names and paths.
        """
        self.slice_index = slice_index
        self.extraction = extraction
        super().__init__(
            caps_directory=caps_directory,
            data_df=data_df,
            extraction=extraction,
            preprocessing=preprocessing,
            transforms=transforms,
        )

    @property
    def elem_index(self):
        return self.slice_index

    def __getitem__(self, idx):
        participant, session, cohort, slice_idx = self._get_meta_data(idx)
        slice_idx = slice_idx + self.extraction.discarded_slices[0]
        image_path = self._get_image_path(participant, session, cohort)

        if self.extraction.save_features:
            slice_dir = image_path.parent.as_posix().replace(
                SubFolder.IMAGE.value, SubFolder.SLICE.value
            )
            slice_filename = self.extraction.extract_slice_path(
                image_path,
                slice_idx,
            )
            slice_tensor = torch.load(
                Path(slice_dir) / slice_filename, weights_only=True
            )

        else:
            image_path = self._get_image_path(participant, session, cohort)
            image = torch.load(image_path, weights_only=True)
            slice_tensor = self.extraction.extract_slice_tensor(
                image,
                slice_idx,
            )

        train_trf, trf = self.transforms.get_transforms()

        slice_tensor = trf(slice_tensor)

        if self.transforms.train_transformations and not self.eval_mode:
            slice_tensor = train_trf(slice_tensor)

        sample = CapsDatasetOutput(
            image=slice_tensor,
            label=label,
            participant_id=participant,
            session_id=session,
            image_id=slice_idx,
            mode=ExtractionMethod.SLICE,
        )

        return sample

    def num_elem_per_image(self):
        if self.elem_index is not None:
            return 1

        if self.extraction.num_slices is not None:
            return self.extraction.num_slices

        image = self._get_full_image()
        return (
            image.size(int(self.extraction.slice_direction) + 1)
            - self.extraction.discarded_slices[0]
            - self.extraction.discarded_slices[1]
        )
