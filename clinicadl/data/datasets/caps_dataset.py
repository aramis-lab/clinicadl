# coding: utf8
from __future__ import annotations

from copy import deepcopy
from logging import getLogger
from pathlib import Path
from typing import List, Optional, Tuple, Union

import nibabel as nib
import pandas as pd
import torch
from joblib import Parallel, delayed
from pydantic import NonNegativeInt, PositiveInt
from torch import save as save_tensor
from torch.utils.data import Dataset
from tqdm import tqdm
from typing_extensions import Self

from clinicadl.data.preprocessing import BasePreprocessing
from clinicadl.data.readers import CapsReader
from clinicadl.data.utils import (
    CapsDatasetSample,
    check_df,
    get_infos_from_json,
    tsv_to_df,
)
from clinicadl.transforms.extraction import Image
from clinicadl.transforms.transforms import Transforms
from clinicadl.utils.exceptions import (
    ClinicaDLCAPSError,
    ClinicaDLConfigurationError,
    ClinicaDLTSVError,
)
from clinicadl.utils.iotools.clinica_utils import create_subs_sess_list

logger = getLogger("clinicadl.caps_dataset")

PARTICIPANT_ID = "participant_id"
SESSION_ID = "session_id"


class CapsDataset(Dataset):
    """
    CapsDataset is a custom PyTorch Dataset class for working with neuroimaging data in CAPS format.

    The dataset supports preprocessing, data augmentation, extraction of specific image
    features (e.g., slices, patches, ROIs), and parallelized preparation of tensor files.

    Parameters
    ----------
        caps_reader: CapsReader
            Reader object for handling CAPS directories.
        preprocessing: BasePreprocessing
            Configuration of preprocessing applied to the data.
        transforms: Transforms
            Transformation pipeline to apply to the data.
        df: pd.DataFrame
            DataFrame containing participant/session information.
        elem_per_image: int
            Number of elements per image, determined by the extraction mode.
        eval_mode: bool
            Flag indicating whether the dataset is in evaluation mode.
    """

    def __init__(
        self,
        caps_directory: Path,
        preprocessing: BasePreprocessing,
        transforms: Transforms,
        data: Optional[Union[pd.DataFrame, Path]] = None,
    ):
        """
        Initializes the CapsDataset.

        Parameters
        ----------
        caps_directory : Path
            Path to the CAPS directory containing the neuroimaging data.
        preprocessing : BasePreprocessing
            Configuration for the preprocessing steps applied to the data.
        transforms : Transforms
            Transformation pipeline to apply to the data during loading.
        data : Union[pd.DataFrame, Path], optional
            Data source, either a TSV file or a pre-loaded DataFrame with participant/session information.
        """

        self.eval_mode = False
        self.caps_reader = CapsReader(caps_directory)
        self.preprocessing = preprocessing
        self.transforms = transforms
        self.extraction = transforms.extraction
        self.df = self._get_df_from_input(data)

        # self.size = self[0].elem.size()

    @property
    def elem_per_image(self):
        """
        Returns the number of elements per image based on the extraction mode.

        The value is determined by extracting the first image in the dataset and checking how many
        elements are present in that image according to the extraction method.

        Returns
        -------
        int
            Number of elements per image.
        """
        if not hasattr(self, "_elem_per_image"):
            self._elem_per_image = self.extraction.num_samples_per_image(
                image=self._get_full_image()[0]
            )
        return self._elem_per_image

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
            "total_samples": self.__len__(),
            "elem_per_image": self._elem_per_image,
            "participants": self.df[PARTICIPANT_ID].nunique(),
            "sessions": self.df[SESSION_ID].nunique(),
            "preprocessing": self.preprocessing.model_dump(),
            "extraction": self.extraction.model_dump(),
        }

    def _get_df_from_input(
        self, data: Optional[Union[pd.DataFrame, Path]] = None
    ) -> pd.DataFrame:
        """
        Generates or validates the DataFrame from the input data.

        Parameters
        ----------
        data : Union[pd.DataFrame, Path], optional
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

        if not self._check_preprocessing_config():
            raise ClinicaDLCAPSError(
                f"The DataFrame does not match the preprocessing configuration: {self.preprocessing.preprocessing.value}"
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
        if isinstance(data, pd.DataFrame):
            df = check_df(data)

        return df

    def _check_preprocessing_config(self) -> bool:
        """
        Validates that the preprocessing configuration matches the data.

        Returns
        -------
        bool
            True if the configuration is valid, otherwise raises an error.

        Raises
        ------
        ClinicaDLConfigurationError
            If the preprocessing configuration does not match the data.
        """
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
        Computes the total number of samples in the dataset.

        Returns
        -------
        int
            Total number of elements in the dataset.
        """
        return len(self.df) * self.elem_per_image

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
            - elem_index (NonNegativeInt): Index of the extracted element.

        Raises
        ------
        IndexError
            If the index is out of range.
        """
        if idx >= self.__len__():
            raise IndexError(
                f"Index out of range, there are only {self.__len__()} elements in your dataset."
            )

        img_idx = idx // self.elem_per_image
        elem_idx = idx % self.elem_per_image

        participant = self._get_participant(idx)
        session = self._get_session(idx)

        return participant, session, img_idx, elem_idx

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

    def _get_participants_sessions_couple(self) -> List[Tuple[str, str]]:
        """
        Retrieves all participant-session pairs in the dataset.

        Returns
        -------
        List[Tuple[str, str]]
            A list of tuples where each tuple contains a participant ID and a session ID.
        """
        return list(zip(self.df[PARTICIPANT_ID], self.df[SESSION_ID]))

    def _get_full_image(
        self, idx: NonNegativeInt = 0, weights_only: bool = True
    ) -> tuple[torch.Tensor, Path]:
        """
        Retrieves the full image tensor and its path for a given index.

        Parameters
        ----------
        idx : NonNegativeInt, optional
            Index of the image (default is 0).
        weights_only : bool, optional
            If True, only the tensor's data weights are loaded (default is True).

        Returns
        -------
        tuple
            A tuple containing:
            - torch.Tensor: The full image tensor.
            - Path: The path to the image file.

        Raises
        ------
        FileNotFoundError
            If the image file does not exist in the CAPS directory.
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

    def __getitem__(self, idx: NonNegativeInt) -> CapsDatasetSample:
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
        """

        if not isinstance(idx, int) or idx < 0:
            raise ValueError(f"Index must be a non-negative integer, got {idx}.")

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
            tensor = self.transforms.extraction.extract_sample(
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

        sample = CapsDatasetSample(
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
        """
        Sets the dataset to evaluation mode.

        This disables data augmentation in the transformation pipeline.

        Returns
        -------
        CapsDataset
            The dataset instance with evaluation mode enabled.
        """
        self.eval_mode = True
        return self

    def train(self):
        """
        Sets the dataset to training mode.

        This enables data augmentation in the transformation pipeline.

        Returns
        -------
        CapsDataset
            The dataset instance with training mode enabled.
        """
        self.eval_mode = False
        return self

    def prepare_data(
        self,
        n_proc: PositiveInt = 2,
        use_uncropped_images: bool = False,
    ):
        """
        Prepares tensor files from the neuroimaging data.

        This method processes the raw neuroimaging data (NIfTI format) into PyTorch tensors
        and stores them for faster data loading during training and evaluation.

        Parameters
        ----------
        n_proc : PositiveInt, optional
            Number of processes to use for parallelization (default is 2).
        use_uncropped_images : bool, optional
            Whether to use uncropped images during preprocessing (default is False).

        Notes
        -----
        - If the tensor file for a participant/session already exists, it will not be reprocessed.
        - This method saves tensor files and image statistics (mean, std, min, max) for each image.
        """

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
            for participant, session in tqdm(
                self._get_participants_sessions_couple(), desc="Preparing data"
            )
        )

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
