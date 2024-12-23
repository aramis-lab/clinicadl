# coding: utf8
from __future__ import annotations

from copy import deepcopy
from logging import getLogger
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import torch
from pydantic import NonNegativeInt
from torch.utils.data import Dataset

from clinicadl.data.preprocessing import Preprocessing, PreprocessingT1
from clinicadl.data.readers.caps_reader import CapsReader
from clinicadl.data.utils import (
    Mask,
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
from clinicadl.utils.typing import DataType, PathType

logger = getLogger("clinicadl.caps_dataset")

PARTICIPANT_ID = "participant_id"
SESSION_ID = "session_id"


class Column(str):
    """
    Dummy class to store label when it represents a column of a dataframe.
    """


class CapsDataset(Dataset):
    """
    Public Methods
    --------------
    from_json : to create a CapsDataset from a configuration file.
    describe : to get a description of the dataset.
    train : to switch the dataset to training mode (i.e. data augmentation is activated).
    eval : to switch the dataset to evaluation mode (i.e. data augmentation is deactivated).
    subset : to get a subset of the CapsDataset (i.e. filtering on (subject, session)
        pairs).
    get_sample_info : to get information on a sample (e.g. the sex or the age of the corresponding
        subject).

    Public attributes
    -----------------
    caps_reader : CapsReader
        Reader object for handling the CAPS directory.
    preprocessing : Preprocessing
        Configuration of preprocessing applied to the data.
    extraction : Extraction
        The extraction object used to get the samples.
    image_transform : Transform
        Transforms that applies on images.
    sample_transform : Transform
        Transforms that applies on samples.
    image_augmentation : Transform
        Augmentations that applies on images.
    sample_augmentation : Transform
        Augmentations that applies on samples.
    df : pd.DataFrame
        DataFrame containing participant/session information.
    label : Optional[Column, Mask]
        The label. Either None, or refers to a column of `self.df` or is
        a `Mask` object.
    masks : Optional[dict[str, Mask]]
        Potential masks, represented by `Mask` objects.
    samples_per_image : int
        Number of samples per image, determined by the extraction mode.
    eval_mode: bool
        Flag indicating whether the dataset is in evaluation mode.
    """

    def __init__(
        self,
        caps_directory: PathType,
        preprocessing: Preprocessing = PreprocessingT1(),
        transforms: Transforms = Transforms(),
        data: Optional[DataType] = None,
        label: Optional[str] = None,
        masks: Optional[dict[str, Union[str, PathType]]] = None,
    ):
        """
        CapsDataset is a custom PyTorch Dataset class for working with neuroimaging data in CAPS format.

        The dataset supports preprocessing, data augmentation, extraction of specific image
        features (e.g., slices or patches), and parallelized preparation of tensor files.

        Parameters
        ----------
        caps_directory : PathType
            Path to the CAPS directory containing the neuroimaging data. A string or a Path object.
        preprocessing : Preprocessing, (optional, default=PreprocessingT1())
            Description of the preprocessing steps applied to the data. Default is Clinica's `t1-linear`
            pipeline.
        transforms : Transforms, (optional, default=Transforms())
            Transformation pipeline to apply to the data during loading. Default will only apply `NaN` removal
            to images.
        data : Optional[DataType], (optional, default=None)
            A DataFrame (or a path to a TSV file containing the dataframe) with the list of subject/session
            pairs to consider, as well as any other relevant information (e.g. the labels for classification or
            regression).\n
            Only subject/session pairs in this TSV file will be in the CapsDataset.\n
            If None, all subject/session pairs in `caps_directory` will be used. Besides, a TSV file
            named `subjects_sessions_list.tsv` will be created in `caps_directory`, with the list of all subject/session
            pairs in the directory.
            .. warning::
                If a `subjects_sessions_list.tsv` already exists in `caps_directory`, it will be overwritten when `data`
                is None.
        label : Optional[str], (optional, default=None)
            A potential label related to the image.\n
            If 'label' and 'data` are not None, CapsDataset will look for a column with that name in 'data'.
            It expects to find the associated column, with floats (regression) or integers (classification).\n
            If there is no such column in 'data' (or if 'data' is None), CapsDataset will look for
            masks with that label as a suffix. The label is thus a mask (segmentation). For example, if
            the image of the subject 'sub-001' for the session 'ses-M000' is in
            'sub-001/ses-M000/sub-001_ses-M000_T1w.nii.gz' and `label="seg"`, it will look for the associated
            mask in 'sub-001/ses-M000/sub-001_ses-M000_seg.nii.gz'.\n
            If None, no label will be used (e.g. for reconstruction).
        masks : Optional[dict[str, Union[str, PathType]]], (optional, default=None)
            Potential masks that are useful to compute some transforms (the values of the dictionary), and their names
            (the keys). A mask can be either a suffix (image-specific masks) or a complete path (common masks).\n
            For example, if `masks={"brain": "brainmask", "hippocampus": "masks/hippocampus.nii.gz"}:
            - For the mask `"hippocampus"`, a path is passed. Therefore, it is understood as a mask common
            to all images. So, CapsDataset will simply get the mask in that file.\n
            - For `"brain"`, a suffix is passed. Therefore, it is understood as an image-specific mask:
            if the image is in 'sub-001/ses-M000/sub-001_ses-M000_T1w.nii.gz', it will look for the masks in
            'sub-001/ses-M000/sub-001_ses-M000_brainmask.nii.gz'.\n

            The names of the masks (`"brain"` and `"hippocampus"`) are used to mention the masks in the transforms (see
            example).

        Raises
        ------
        ValueError
            If 'data' is not a DataFrame, a path or None.
        ClinicaDLTSVError
            If 'data' is a TSV file that does not exist.
        ClinicaDLTSVError
            If the DataFrame associated to `data` does not contain the columns `"participant_id"`
            and `"session_id"`.
        ClinicaDLConfigurationError
            If the data does not match the preprocessing configuration.
        ValueError
            If 'label' is not a string or None.
        FileNotFoundError
            If `masks` contain paths that does not match any files.

        Examples
        --------
        >>>
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
        self.masks = (
            {name: Mask(mask) for name, mask in masks.items()}
            if masks is not None
            else None
        )
        self._samples_per_image = None
        self._image_shape = None

    @property
    def samples_per_image(self) -> NonNegativeInt:
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
        # TODO : review this function and write to associated `write_to_json``

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

    def describe(self) -> dict[str, Any]:
        """
        Returns a description of the CapsDataset.

        Returns
        -------
        dict[str, Any]
            A dictionary containing:
            - `total_samples`: the size of the dataset, i.e.
            the number of images times the number of samples per
            image.
            - `samples_per_image`: the number of samples per image.
            - `participant_session_pairs`: the list of subject/session
            pairs in the dataset.
            - `preprocessing`: the preprocessing parameters.
            - `extraction`: the extraction parameters.
        """
        return {
            "total_samples": len(self),
            "samples_per_image": self._samples_per_image,
            "participant_session_pairs": self._get_participant_session_couples(),
            "preprocessing": self.preprocessing.model_dump(),
            "extraction": self.extraction.model_dump(),
        }

    def eval(self) -> None:
        """
        Sets the dataset to evaluation mode.

        This disables data augmentation in the transformation pipeline.
        """
        self.eval_mode = True

    def train(self) -> None:
        """
        Sets the dataset to training mode.

        This enables data augmentation in the transformation pipeline.
        """
        self.eval_mode = False

    def subset(self, data: DataType) -> CapsDataset:
        """
        To get a subset of the CapsDataset from a list of (subject, session) pairs.

        Parameters
        ----------
        data : DataType
            A DataFrame (or a path to a TSV file containing the dataframe) with the list of subject/session
            pairs to extract. Please note that this list must be passed via two columns named `"participant_id`"
            and `"session_id"` (other columns won't be considered).

        Returns
        -------
        CapsDataset
            A subset of the original CapsDataset, restricted to the (subject, session) pairs mentioned in `data`.

        Raises
        ------
        ClinicaDLTSVError
            If `data` is a TSV file that does not exist.
        ClinicaDLTSVError
            If the DataFrame associated to `data` does not contain the columns `"participant_id"`
            and `"session_id"`.
        ClinicaDLTSVError
            If some (participant_id, session_id) pairs mentioned in `data` are not in the CapsDataset.
        """
        df = self._check_data_instance(data).set_index([PARTICIPANT_ID, SESSION_ID])

        try:
            subset_df = (
                self.df.set_index([PARTICIPANT_ID, SESSION_ID])
                .loc[df.index]
                .reset_index()
            )
        except KeyError as exc:
            missing_rows = df.index.difference(
                self.df.set_index([PARTICIPANT_ID, SESSION_ID]).index
            )

            err_message = "Missing rows: \n"
            for row in missing_rows:
                err_message += f" - {row} \n"

            raise ClinicaDLTSVError(
                "Some couples (participant_id, session_id) are not in the dataset,",
                err_message,
            ) from exc

        dataset = deepcopy(self)
        dataset.df = subset_df

        return dataset

    def get_sample_info(self, idx: NonNegativeInt, column: str) -> Any:
        """
        Retrieves information on a given sample. The information will
        correspond to the information on the base image the sample was extracted
        from.

        Parameters
        ----------
        idx : NonNegativeInt
            The index of the sample in the dataset.
        column : str
            The information to look for, i.e. a column of the DataFrame containing
            the metadata, which is equal to `data` if `data` was passed when instantiating the
            CapsDataset. If `data` was not passed, the only accessible columns are
            `"participant_id"` and `"session_id"`.

        Returns
        -------
        Any
            the information (e.g. the age, the sex, etc.)

        Raises
        ------
        IndexError
            If 'idx' is greater or equal to the length of the dataset.
        KeyError
            If `column` is not in the metadata DataFrame.
        """
        if idx >= len(self):
            raise IndexError(
                f"Index out of range, there are only {len(self)} samples in the dataset."
            )
        if column not in self.df.columns:
            raise KeyError(
                f"No column named {column} in the metadata DataFrame. Present columns are: "
                f"{list(self.df.columns)}"
            )

        img_idx = idx // self.samples_per_image
        return self.df.at[img_idx, column]

    def __len__(self) -> NonNegativeInt:
        """
        Computes the total number of samples in the dataset.

        Returns
        -------
        NonNegativeInt
            Total number of samples in the dataset, i.e. the number of images
            times the number of samples per image.
        """
        return len(self.df) * self.samples_per_image

    def __getitem__(self, idx: NonNegativeInt) -> Sample:
        """
        Retrieves the sample at a given index.

        Parameters
        ----------
        idx : NonNegativeInt
            Index of the sample.

        Returns
        -------
        Sample
            A structured output containing the processed data and metadata.

        Raises
        ------
        ValueError
            If 'idx' is not an non-negative integer.
        IndexError
            If 'idx' is greater or equal to the length of the dataset.
        ClinicaDLCAPSError
            If there is no or more than one images associated to the (subject, session) in the
            CAPS directory.
        ClinicaDLCAPSError
            If the label cannot be found, neither in the columns of `data`, nor in `.pt` file,
            nor in a NIfTI file.
        ClinicaDLCAPSError
            If a potential mask cannot be found, neither in `.pt` nor in a NIfTI file.
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

    def _check_label(self, label: Optional[str]) -> Optional[Union[Column, Mask]]:
        """
        Checks if 'label' is a column name, a mask suffix or None.

        Raises
        ------
        ValueError
            If 'label' is not a string or None.
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

    def _get_df_from_input(self, data: Optional[DataType]) -> pd.DataFrame:
        """
        Generates or validates the DataFrame from the input data.

        Raises
        ------
        ValueError
            If 'data' is not a DataFrame, a path or None.
        ClinicaDLTSVError
            If 'data' is a TSV file that does not exist.
        ClinicaDLConfigurationError
            If the data does not match the preprocessing configuration.
        """

        if data is None:
            data = create_subs_sess_list(
                self.caps_reader.input_directory,
                self.caps_reader.input_directory,
                is_bids_dir=False,
            )
            logger.info(f"Creating a subject session TSV file at {data}")

        if not isinstance(data, DataType):
            raise ValueError(
                f"'data' must be a Pandas DataFrame, a path to a TSV file or None. Got {data}"
            )

        df = self._check_data_instance(data)
        self.df = df
        self.caps_reader.check_preprocessing(
            self._get_participant_session_couples(), self.preprocessing
        )

        return df

    @staticmethod
    def _check_data_instance(data: DataType) -> pd.DataFrame:
        """
        Checks the DataFrame passed by the user (either as a DataFrame or
        as a path to a TSV). Returns the checked DataFrame.

        Raises
        ------
        ClinicaDLTSVError
            If 'data' is a TSV file that does not exist.
        ClinicaDLTSVError
            If the DataFrame does not contain the columns `"participant_id"`
            and `"session_id"`.
        """

        if isinstance(data, PathType):
            data = Path(data)
            try:
                df = tsv_to_df(data)
            except ClinicaDLTSVError as exc:
                raise ClinicaDLTSVError(
                    f"The 'data' file does not exist: {data}"
                    "Please ensure the file path is correct and accessible."
                ) from exc
        elif isinstance(data, pd.DataFrame):
            df = check_df(data)

        return df  # pylint: disable=possibly-used-before-assignment

    def _get_meta_data(
        self, idx: NonNegativeInt
    ) -> Tuple[str, str, NonNegativeInt, NonNegativeInt]:
        """
        Retrieves metadata for a given sample index.

        Returns
        -------
        tuple
            - participant (str): ID of the participant.
            - session (str): ID of the session.
            - img_index (NonNegativeInt): index of the image.
            - sample_index (NonNegativeInt): index of the extracted sample.

        Raises
        ------
        IndexError
            If the index is out of range.
        """
        participant = self.get_sample_info(idx, PARTICIPANT_ID)
        session = self.get_sample_info(idx, SESSION_ID)
        img_idx = idx // self.samples_per_image
        sample_idx = idx % self.samples_per_image

        return participant, session, img_idx, sample_idx

    def _get_participant(self, img_idx: NonNegativeInt) -> str:
        """
        Retrieves the participant ID for a given image index.
        """
        return self.df.at[img_idx, PARTICIPANT_ID]

    def _get_session(self, img_idx: NonNegativeInt) -> str:
        """
        Retrieves the session ID for a given image index.
        """
        return self.df.at[img_idx, SESSION_ID]

    def _get_participant_session_couples(self) -> List[Tuple[str, str]]:
        """
        Retrieves all participant-session pairs in the dataset.
        """
        return list(zip(self.df[PARTICIPANT_ID], self.df[SESSION_ID]))

    def _get_full_image(self, img_idx: NonNegativeInt) -> tuple[torch.Tensor, Path]:
        """
        Retrieves the full image tensor and its path for a given image index.
        Will first look for a `.pt` file. If not found, will look for a NIfTI file.

        Returns
        -------
        tuple
            A tuple containing:
            - torch.Tensor: The full image, as a 4D tensor (including one channel dimension).
            - Path: The path to the image file.

        Raises
        ------
        ClinicaDLCAPSError
            If there is no or more than one images associated to the (subject, session) in the
            CAPS directory.
        """

        participant_id = self._get_participant(img_idx)
        session_id = self._get_session(img_idx)

        pt_image_path = self.caps_reader.get_tensor_path(
            participant_id, session_id, self.preprocessing
        )
        if pt_image_path.is_file():
            return pt_to_tensor(pt_image_path), pt_image_path

        nifti_image_path = self.caps_reader.get_image_path(
            participant_id, session_id, self.preprocessing
        )
        return nifti_to_tensor(nifti_image_path), nifti_image_path

    def _get_single_mask(self, img_idx: NonNegativeInt, mask: Mask) -> torch.Tensor:
        """
        Retrieves a mask associated to an image, from the index of that image
        and from the Mask object.

        Will first look for the mask in a `.pt` file, then in a NIfTI file.

        Raises
        ------
        ClinicaDLCAPSError
            If there is no or more than one images associated to the (subject, session) in the
            CAPS directory.
        ClinicaDLCAPSError
            If associated mask cannot be found, neither in `.pt` nor in a NIfTI file.
        """
        participant_id = self._get_participant(img_idx)
        session_id = self._get_session(img_idx)

        pt_image_path = self.caps_reader.get_tensor_path(
            participant_id, session_id, self.preprocessing
        )
        try:
            return mask.get_associated_mask(pt_image_path)
        except FileNotFoundError:
            nifti_image_path = self.caps_reader.get_image_path(
                participant_id, session_id, self.preprocessing
            )
            try:
                return mask.get_associated_mask(nifti_image_path)
            except FileNotFoundError as exc:
                raise ClinicaDLCAPSError(
                    f"Cannot find the mask '{mask.mask}' associated to subject={participant_id} "
                    f"and session={session_id}. The mask was expected in "
                    f"{mask.get_associated_mask_path(pt_image_path)} or {mask.get_associated_mask_path(nifti_image_path)}."
                ) from exc

    def _get_masks(self, idx: NonNegativeInt) -> Dict[str, torch.Tensor]:
        """
        Retrieves all the masks associated to an image from the index of that image.

        Raises
        ------
        ClinicaDLCAPSError
            If there is no or more than one images associated to the (subject, session) in the
            CAPS directory.
        ClinicaDLCAPSError
            If associated mask cannot be found, neither in `.pt` nor in a NIfTI file.
        """
        if self.masks is not None:
            return {
                name: self._get_single_mask(idx, mask)
                for name, mask in self.masks.items()
            }
        else:
            return {}

    def _get_label(
        self, img_idx: NonNegativeInt
    ) -> Optional[Union[int, float, torch.Tensor]]:
        """
        Retrieves the label associated to an image from the index of that image.

        Raises
        ------
        ClinicaDLCAPSError
            If the label cannot be found, neither in the columns of `data`, nor in `.pt` file,
            nor in a NIfTI file.
        """
        if isinstance(self.label, Column):
            return self.df.at[img_idx, self.label]
        elif isinstance(self.label, Mask):
            try:
                return self._get_single_mask(img_idx, self.label)
            except ClinicaDLCAPSError as exc:
                raise ClinicaDLCAPSError(
                    f"No column named {self.label} in 'data', so label={self.label} is "
                    f"understood as a file suffix. But no file found for subject={self._get_participant(img_idx)} "
                    f"and session={self._get_session(img_idx)}."
                ) from exc
        else:
            return None
