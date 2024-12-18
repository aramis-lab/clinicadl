# coding: utf8
from __future__ import annotations

from copy import deepcopy
from logging import getLogger
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import torch
from pydantic import NonNegativeInt
from torch.utils.data import Dataset

from clinicadl.data.preprocessing import Preprocessing
from clinicadl.data.readers.caps_reader import CapsReader
from clinicadl.data.utils import (
    check_df,
    get_infos_from_json,
    tsv_to_df,
)
from clinicadl.transforms.extraction import Sample
from clinicadl.transforms.transforms import Transforms
from clinicadl.transforms.utils import get_tio_image
from clinicadl.utils.exceptions import ClinicaDLCAPSError, ClinicaDLTSVError
from clinicadl.utils.iotools.clinica_utils import create_subs_sess_list
from clinicadl.utils.loading import nifti_to_tensor, pt_to_tensor

logger = getLogger("clinicadl.caps_dataset")

PARTICIPANT_ID = "participant_id"
SESSION_ID = "session_id"


class Column(str):
    """
    Dummy class to store label when it represents a column of a dataframe.
    """


class Mask(str):
    """
    Dummy class to store label when it represents the suffix of a mask.
    """


class CapsDataset(Dataset):
    """
    CapsDataset is a custom PyTorch Dataset class for working with neuroimaging data in CAPS format.

    The dataset supports preprocessing, data augmentation, extraction of specific image
    features (e.g., slices, patches, ROIs), and parallelized preparation of tensor files.

    Parameters
    ----------
        caps_reader: CapsReader
            Reader object for handling CAPS directories.
        preprocessing: Preprocessing
            Configuration of preprocessing applied to the data.
        transforms: Transforms
            Transformation pipeline to apply to the data.
        df: pd.DataFrame
            DataFrame containing participant/session information.
        sample_per_image: int
            Number of samples per image, determined by the extraction mode.
        eval_mode: bool
            Flag indicating whether the dataset is in evaluation mode.
    """

    def __init__(
        self,
        caps_directory: Union[str, Path],
        preprocessing: Preprocessing,
        transforms: Transforms = Transforms(),
        data: Optional[Union[pd.DataFrame, str, Path]] = None,
        label: Optional[str] = None,
        masks: Optional[list[str]] = None,
    ):
        """
        Initializes the CapsDataset.

        Parameters
        ----------
        caps_directory : Path
            Path to the CAPS directory containing the neuroimaging data.
        preprocessing : Preprocessing
            Configuration for the preprocessing steps applied to the data.
        transforms : Transforms
            Transformation pipeline to apply to the data during loading.
        data : Union[pd.DataFrame, Path], (optional, default=None)
            Data source, either a TSV file or a pre-loaded DataFrame with participant/session information.
            Only subject/session pairs in this TSV file will be in the CapsDataset.\n
            If None, all subject/session pairs in `caps_directory` will be used. Besides, a TSV file
            named `subjects_sessions_list.tsv` will be created in `caps_directory`, with the list of all subject/session
            pairs in the directory.
            .. warning::
                If a `subjects_sessions_list.tsv` already exists in `caps_directory`, it will be overwritten when `data`
                is None.
        label : Optional[str] (optional, default=None)
            A potential label related to the image.\n
            If 'label' is not None, CapsDataset will look for a column with that name in 'data'.
            It expects to find a column with floats (regression) or integers (classification).\n
            If there is no such column in 'data' (or if 'data' is None), CapsDataset will look for
            masks with that label as a suffix. The label is thus a mask (segmentation). For example, if
            the image of the subject 'sub-001' for the session 'ses-M000' is in
            'sub-001/ses-M000/sub-001_ses-M000_T1w.nii.gz' and `label="seg"`, it will look for the associated
            mask in 'sub-001/ses-M000/sub-001_ses-M000_seg.nii.gz'.\n
            If None, no label will be used (e.g. for reconstruction).
        masks : Optional[list[str]] (optional, default=None)
            Potential subject-specific masks that are useful to compute some transforms. CapsDataset
            will look for masks with that values as a suffix. For example, if the image of the subject
            'sub-001' for the session 'ses-M000' is in 'sub-001/ses-M000/sub-001_ses-M000_T1w.nii.gz'
            and `masks=["brain", "hippocampus"]`, it will look for the masks in
            'sub-001/ses-M000/sub-001_ses-M000_brain.nii.gz' and 'sub-001/ses-M000/sub-001_ses-M000_hippocampus.nii.gz'.
        """

        self.eval_mode = False
        self.caps_reader = CapsReader(caps_directory)
        self.preprocessing = preprocessing
        (
            self.image_transform,
            self.sample_transform,
            self.image_augmentation,
            self.sample_augmentation,
        ) = transforms.get_transforms()
        self.extraction = transforms.extraction
        self.df = self._get_df_from_input(data)
        self.label = self._check_label(label)
        self.masks = masks
        self._samples_per_image = None
        self._image_shape = None

    def _check_label(self, label: Optional[str]) -> Optional[Union[Column, Mask]]:
        """
        Checks if 'label' is a column name, a mask suffix or None.
        """
        if isinstance(label, str):
            if label in self.df.columns:
                return Column(label)
            else:
                return Mask(label)
        elif label is None:
            return None
        else:
            raise ValueError(f"'label' must be a string or None. Got {label}")

    @property
    def samples_per_image(self) -> int:
        """
        Returns the number of samples per image based on the extraction mode.

        The value is determined by extracting the first image in the dataset and checking how many
        samples are present in that image according to the extraction method.
        """
        if self._samples_per_image is None:
            image: torch.Tensor = self._get_full_image(0)[0]
            self._samples_per_image = self.extraction.num_samples_per_image(image)
            self._image_shape = tuple(image.shape)

        return self._samples_per_image

    @classmethod
    def from_json(cls, json_path: Path):
        """
        Creates a CapsDataset instance from a JSON configuration file.

        This method loads the preprocessing configuration, transformation pipeline, CAPS directory,
        and data source (TSV or DataFrame) from the provided JSON file, and returns an instance
        of the CapsDataset.

        Parameters
        ----------
        json_path : Path
            Path to the JSON file containing the necessary configuration for creating the dataset.

        Returns
        -------
        CapsDataset
            The initialized CapsDataset instance.

        Raises
        ------
        FileNotFoundError
            If the provided JSON file does not exist.
        """

        if not json_path.is_file():
            raise FileNotFoundError(
                f"The provided preprocessing JSON file {json_path} does not exist."
            )

        preprocessing, transforms, caps_dir, data_tsv = get_infos_from_json(json_path)
        return CapsDataset(
            caps_dir,
            preprocessing,
            transforms,
            data_tsv,
        )

    def describe(self):
        """To complete/merge later with the dataset_description from clinica"""
        return {
            "total_samples": len(self),
            "samples_per_image": self._samples_per_image,
            "participants": self.df[PARTICIPANT_ID].nunique(),
            "sessions": self.df[SESSION_ID].nunique(),
            "preprocessing": self.preprocessing.model_dump(),
            "extraction": self.extraction.model_dump(),
        }

    def _get_df_from_input(
        self, data: Optional[Union[pd.DataFrame, Path]]
    ) -> pd.DataFrame:
        """
        Generates or validates the DataFrame from the input data.

        Parameters
        ----------
        data : Union[pd.DataFrame, Path]
            Path to the TSV file or a DataFrame containing participant/session pairs.

        Returns
        -------
        pd.DataFrame
            Validated DataFrame containing participant/session information.

        Raises
        ------
        ClinicaDLTSVError
            If the provided TSV file does not exist.
        ClinicaDLCAPSError
            If the data does not match the preprocessing configuration.
        """

        if data is None:
            data = create_subs_sess_list(
                self.caps_reader.input_directory,
                self.caps_reader.input_directory,
                is_bids_dir=False,
            )
            logger.info(f"Creating a subject session TSV file at {data}")

        df = self._check_data_instance(data)
        self.df = df

        self.caps_reader.check_preprocessing(
            self._get_participant_session_couples(), self.preprocessing
        )

        return df

    def _check_data_instance(self, data: Optional[Union[pd.DataFrame, Path]] = None):
        if isinstance(data, str):
            data = Path(data)

        if isinstance(data, Path):
            if not data.is_file():
                raise ClinicaDLTSVError(
                    f"The data file does not exist: {data}"
                    "Please ensure the file path is correct and accessible."
                )
            df = tsv_to_df(data)
        elif isinstance(data, pd.DataFrame):
            df = check_df(data)
        else:
            raise ValueError(
                f"'data' must be a Pandas DataFrame, a path to a TSV file or None. Got {data}"
            )

        return df

    def __len__(self) -> int:
        """
        Computes the total number of samples in the dataset.

        Returns
        -------
        int
            Total number of samples in the dataset.
        """
        return len(self.df) * self.samples_per_image

    def _get_meta_data(
        self, idx: NonNegativeInt
    ) -> Tuple[str, str, NonNegativeInt, NonNegativeInt]:
        """
        Retrieves metadata for a given sample index.

        Parameters
        ----------
        idx : NonNegativeInt
            Index of the sample.

        Returns
        -------
        tuple
            - participant (str): ID of the participant.
            - session (str): ID of the session.
            - img_index (NonNegativeInt): Index of the image.
            - sample_index (NonNegativeInt): Index of the extracted sample.

        Raises
        ------
        IndexError
            If the index is out of range.
        """
        if idx >= len(self):
            raise IndexError(
                f"Index out of range, there are only {len(self)} samples in your dataset."
            )

        img_idx = idx // self.samples_per_image
        sample_idx = idx % self.samples_per_image

        participant = self._get_participant(idx)
        session = self._get_session(idx)

        return participant, session, img_idx, sample_idx

    def _get_participant(self, idx: NonNegativeInt) -> str:
        """
        Retrieves the participant ID for a given row index.

        Parameters
        ----------
        idx : NonNegativeInt
            Row index.

        Returns
        -------
        str
            Participant ID.
        """
        return self.df.at[idx, PARTICIPANT_ID]

    def _get_session(self, idx: NonNegativeInt) -> str:
        """
        Retrieves the session ID for a given row index.

        Parameters
        ----------
        idx : NonNegativeInt
            Row index.

        Returns
        -------
        str
            Session ID.
        """

        return self.df.at[idx, SESSION_ID]

    def _get_participant_session_couples(self) -> List[Tuple[str, str]]:
        """
        Retrieves all participant-session pairs in the dataset.

        Returns
        -------
        List[Tuple[str, str]]
            A list of tuples where each tuple contains a participant ID and a session ID.
        """
        return list(zip(self.df[PARTICIPANT_ID], self.df[SESSION_ID]))

    def _get_full_image(self, idx: NonNegativeInt) -> tuple[torch.Tensor, Path]:
        """
        Retrieves the full image tensor and its path for a given index.

        Parameters
        ----------
        idx : NonNegativeInt, optional
            Index of the image.

        Returns
        -------
        tuple
            A tuple containing:
            - torch.Tensor: The full image tensor.
            - Path: The path to the image file.

        Raises
        ------
        ClinicaDLCAPSError
            If there is no image associated to the (subject, session) in the
            CAPS directory.
        """

        participant_id = self._get_participant(idx)
        session_id = self._get_session(idx)

        pt_image_path = self.caps_reader.get_tensor_path(
            participant_id, session_id, self.preprocessing
        )
        if pt_image_path.is_file():
            return pt_to_tensor(pt_image_path), pt_image_path

        nifti_image_path = self.caps_reader.get_image_path(
            participant_id, session_id, self.preprocessing
        )
        return nifti_to_tensor(nifti_image_path), nifti_image_path

    def _get_single_mask(self, idx: int, mask: str) -> torch.Tensor:
        """
        Retrieves a mask associated to an image, from the index of that image
        and from the name of the mask, interpreted as the suffix of the file
        where the mask is stored.
        """
        participant_id = self._get_participant(idx)
        session_id = self._get_session(idx)

        pt_image_path = self.caps_reader.get_tensor_path(
            participant_id, session_id, self.preprocessing
        )
        pt_mask_path = self.caps_reader.replace_suffix(pt_image_path, mask)
        if pt_mask_path.is_file():
            return pt_to_tensor(pt_mask_path, int_values=True)

        nifti_image_path = self.caps_reader.get_image_path(
            participant_id, session_id, self.preprocessing
        )
        nifti_mask_path = self.caps_reader.replace_suffix(nifti_image_path, mask)
        if nifti_mask_path.is_file():
            return nifti_to_tensor(nifti_mask_path, int_values=True)

        raise ClinicaDLCAPSError(
            f"Cannot find a mask with the suffix '{mask}' associated to subject={participant_id} "
            f"and session={session_id}. The mask was expected in {pt_mask_path} or {nifti_mask_path}."
        )

    def _get_masks(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Retrieves all the masks associated to an image.

        Parameters
        ----------
        idx : int
            Index of the image.

        Raises
        ------
        ClinicaDLCAPSError
            If there is no image associated to the (subject, session) in the
            CAPS directory.
        FileNotFoundError
            If a mask is not found.
        """
        if self.masks is not None:
            return {"mask": self._get_single_mask(idx, mask) for mask in self.masks}
        else:
            return {}

    def _get_label(
        self, idx: NonNegativeInt
    ) -> Optional[Union[int, float, torch.Tensor]]:
        """
        Retrieves the label associated to an image for a given index.

        Parameters
        ----------
        idx : NonNegativeInt
            Index of the image.

        Returns
        -------
        Optional[[int, float, torch.Tensor]]
            The associated label.

        Raises
        ------
        FileNotFoundError
            If the label is an image, which was not found in the CAPS directory.
        """
        if isinstance(self.label, Column):
            return self.df.at[idx, self.label]
        elif isinstance(self.label, Mask):
            try:
                return self._get_single_mask(idx, self.label)
            except FileNotFoundError as exc:
                raise FileNotFoundError(
                    f"No column named {self.label} in 'data', so label={self.label} is "
                    f"understood as a file suffix. But no file found for subject={self._get_participant(idx)} "
                    f"and session={self._get_session(idx)}."
                ) from exc
        else:
            return None

    def __getitem__(self, idx: NonNegativeInt) -> Sample:
        """
        Retrieves the sample at a given index.

        Parameters
        ----------
        idx : NonNegativeInt
            Index of the sample.

        Returns
        -------
        CapsDatasetSample
            A structured output containing the processed data and metadata.

        Raises
        ------
        ValueError
            If 'idx' is not an integer.
        IndexError
            If 'idx' is greater or equal to the length of the dataset.
        """

        if not isinstance(idx, int) or idx < 0:
            raise ValueError(f"Index must be a non-negative integer, got {idx}.")

        participant, session, img_index, sample_index = self._get_meta_data(idx)
        image, image_path = self._get_full_image(img_index)
        label = self._get_label(img_index)
        masks = self._get_masks(img_index)

        tio_image = get_tio_image(image, label, **masks)

        tio_image = self.image_transform(tio_image)
        if not self.eval_mode:
            tio_image = self.image_augmentation(tio_image)

        tio_sample, sample_description = self.extraction.extract_tio_sample(
            tio_image, sample_index
        )

        tio_sample = self.sample_transform(tio_sample)
        if not self.eval_mode:
            tio_sample = self.sample_augmentation(tio_sample)

        return self.extraction.format_output(
            tio_sample,
            participant_id=participant,
            session_id=session,
            image_path=image_path,
            description=sample_description,
        )

    def eval(self):
        """
        Sets the dataset to evaluation mode.

        This disables data augmentation in the transformation pipeline.
        """
        self.eval_mode = True

    def train(self):
        """
        Sets the dataset to training mode.

        This enables data augmentation in the transformation pipeline.
        """
        self.eval_mode = False

    def subset(self, data: Optional[Union[pd.DataFrame, Path]] = None) -> CapsDataset:
        df = self._check_data_instance(data)

        common_rows = pd.merge(df, self.df, how="inner")
        all_included = len(common_rows) == len(df)

        if not all_included:
            missing_rows = pd.concat(
                [df, common_rows], ignore_index=True
            ).drop_duplicates(keep=False)

            err_message = "Missing rows: \n"
            for row in missing_rows:
                err_message += f" - {row} \n"

            raise ClinicaDLTSVError(
                "Some couples (participanst_id, session_id) are not in the dataset,",
                err_message,
            )

        dataset = deepcopy(self)
        dataset.df = df

        return dataset
