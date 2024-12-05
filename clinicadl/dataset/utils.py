# coding: utf8
from glob import glob
from logging import getLogger
from pathlib import Path
from typing import List, Optional, Tuple, Union

import pandas as pd
import torch
import torchio as tio
from pydantic import BaseModel, ConfigDict

from clinicadl.dataset import preprocessing
from clinicadl.transforms import extraction
from clinicadl.transforms.transforms import Transforms
from clinicadl.utils.enum import ExtractionMethod, Preprocessing
from clinicadl.utils.exceptions import ClinicaDLTSVError
from clinicadl.utils.iotools.utils import read_preprocessing

logger = getLogger("clinicadl.dataset.utils")

PARTICIPANT_ID = "participant_id"
SESSION_ID = "session_id"


class CapsDatasetSample(BaseModel):
    """
    A data model representing the output from a CapsDataset.

    Args:
        elem (torch.Tensor): The image tensor (processed data).
        participant_id (Union[int, str]): The participant's identifier.
        session_id (Union[int, str]): The session identifier.
        label (Optional[Union[float, int]], optional): The label associated with the data (default is None).
        img_idx (Optional[Union[int, str]], optional): An optional image index (default is None).
        elem_idx (Optional[Union[int, str]], optional): An optional element index (default is None).
        image_path (Optional[Path], optional): The file path to the image (default is None).
        mode (ExtractionMethod): The extraction method used to process the data.

    Attributes:
        model_config (ConfigDict): Configuration options for the model.
    """

    elem: torch.Tensor
    participant_id: Union[int, str]
    session_id: Union[int, str]
    label: Optional[Union[float, int]] = None
    img_idx: Optional[Union[int, str]] = None
    elem_idx: Optional[Union[int, str]] = None
    image_path: Optional[Path] = None
    mode: ExtractionMethod

    model_config = ConfigDict(validate_assignment=True, arbitrary_types_allowed=True)


def tsv_to_df(tsv_path: Path) -> pd.DataFrame:
    """
    Converts a TSV file to a Pandas DataFrame.

    Args:
        tsv_path (Path): Path to the TSV file to be read.

    Returns:
        pd.DataFrame: The resulting DataFrame containing the TSV data.

    Raises:
        ClinicaDLTSVError: If the TSV file cannot be found or is not in the correct format.
    """
    try:
        df = pd.read_csv(tsv_path, sep="\t")
    except FileNotFoundError:
        raise ClinicaDLTSVError(
            "The TSV file you gave is not a file.\nError explanations:\n"
            f"\t- Clinica expected the following path to be a file: {tsv_path}\n"
            "\t- If you gave a relative path, did you run Clinica on the correct folder?"
        )
    df = check_df(df)
    return df


def check_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Checks if the DataFrame contains the required columns: 'participant_id' and 'session_id'.

    Args:
        df (pd.DataFrame): The DataFrame to be checked.

    Returns:
        pd.DataFrame: The same DataFrame if the required columns are present.

    Raises:
        ClinicaDLTSVError: If the required columns ('participant_id', 'session_id') are not found in the DataFrame.
    """
    if not {PARTICIPANT_ID, SESSION_ID}.issubset(set(df.columns.values)):
        raise ClinicaDLTSVError(
            f"The data file is not in the correct format. "
            f"Columns should include {PARTICIPANT_ID, SESSION_ID}"
        )

    return df


def reset_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resets the index of a DataFrame to the default index, dropping any existing index.

    Args:
        df (pd.DataFrame): The DataFrame to be reset.

    Returns:
        pd.DataFrame: The DataFrame with the default index.

    Note:
        This function only resets the index if the DataFrame has a MultiIndex with the 'participant_id' and'session_id' names.
        If the DataFrame does not have this MultiIndex, the 'drop' parameter is set to True, which results in dropping the index.
    """

    drop = False
    if isinstance(df.index, pd.MultiIndex):
        if set(df.index.names) != {PARTICIPANT_ID, SESSION_ID}:
            drop = True

    df.reset_index(inplace=True, drop=drop)

    return df


def insensitive_glob(pattern_glob: str, recursive: bool = False) -> List[str]:
    """
    Perform a case-insensitive glob search.

    Args:
        pattern_glob (str): The pattern to search for, sensitive to case.
        recursive (bool, optional): If True, performs the glob search recursively. Default is False.

    Returns:
        List[str]: A list of matching file paths, case-insensitive to the given pattern.
    """

    def make_case_insensitive_pattern(c: str) -> str:
        """
        Convert a character to a case-insensitive pattern for glob matching.

        Args:
            c (str): The character to be converted.

        Returns:
            str: A case-insensitive pattern for the character (e.g., '[aA]' for 'a').
        """
        return "[%s%s]" % (c.lower(), c.upper()) if c.isalpha() else c

    insensitive_pattern = "".join(map(make_case_insensitive_pattern, pattern_glob))
    return glob(insensitive_pattern, recursive=recursive)


def get_extraction(
    extract_method: Union[str, ExtractionMethod],
) -> type[extraction.Extraction]:
    """
    Retrieves the extraction method based on the specified extraction method.

    Args:
        extract_method (Union[str, ExtractionMethod]): The extraction method as either a string or an `ExtractionMethod` enum.

    Returns:
        type[extraction.Extraction]: The corresponding extraction class (e.g., `ROI`, `Slice`, etc.).

    Raises:
        ValueError: If the provided `extract_method` is not supported or is invalid.
    """
    extract_method = ExtractionMethod(extract_method)
    if extract_method == ExtractionMethod.ROI:
        return extraction.ROI
    elif extract_method == ExtractionMethod.SLICE:
        return extraction.Slice
    elif extract_method == ExtractionMethod.IMAGE:
        return extraction.Image
    elif extract_method == ExtractionMethod.PATCH:
        return extraction.Patch
    else:
        raise ValueError(f"Preprocessing {extract_method.value} is not implemented.")


def get_preprocessing(
    preprocessing_type: Union[str, Preprocessing],
) -> type[preprocessing.BasePreprocessing]:
    """
    Retrieves the preprocessing class based on the specified preprocessing type.

    Args:
        preprocessing_type (Union[str, Preprocessing]): The preprocessing type as either a string or a `Preprocessing` enum.

    Returns:
        type[preprocessing.BasePreprocessing]: The corresponding preprocessing configuration class.

    Raises:
        ValueError: If the provided `preprocessing_type` is not supported or is invalid.
    """
    preprocessing_type = Preprocessing(preprocessing_type)
    if preprocessing_type == Preprocessing.T1_LINEAR:
        return preprocessing.PreprocessingT1
    elif preprocessing_type == Preprocessing.PET_LINEAR:
        return preprocessing.PreprocessingPET
    elif preprocessing_type == Preprocessing.FLAIR_LINEAR:
        return preprocessing.PreprocessingFlair
    elif preprocessing_type == Preprocessing.CUSTOM:
        return preprocessing.PreprocessingCustom
    elif preprocessing_type == Preprocessing.DWI_DTI:
        return preprocessing.PreprocessingDTI
    else:
        raise ValueError(
            f"Preprocessing {preprocessing_type.value} is not implemented."
        )


def get_infos_from_json(
    json_path: Path,
) -> Tuple[preprocessing.BasePreprocessing, Transforms, Path, Path]:
    """
    Extracts the preprocessing configuration and transformation settings from a JSON file.

    Args:
        json_path (Path): The path to the JSON file containing the preprocessing and transformation details.

    Returns:
        Tuple[Preprocessing, Transforms, Path, Path]:
            A tuple containing the preprocessing configuration, transformation settings,
            CAPS directory path, and the path to the participant/session information TSV file.

    Raises:
        ClinicaDLTSVError: If there is an error reading the JSON or the provided paths are incorrect.
    """
    dict_ = read_preprocessing(json_path)
    return get_infos_from_parameters(**dict_)


def get_infos_from_parameters(
    **kwargs,
) -> Tuple[preprocessing.BasePreprocessing, Transforms, Path, Path]:
    """
    Extracts the preprocessing configuration, transformations, and paths from provided parameters.

    Args:
        **kwargs: A set of keyword arguments, expected to contain necessary details like preprocessing type,
                  extraction method, CAPS directory, and TSV file paths.

    Returns:
        Tuple[Preprocessing, Transforms, Path, Path]:
            A tuple containing the preprocessing configuration, transformation settings,
            CAPS directory path, and the path to the participant/session information TSV file.

    Raises:
        ValueError: If required parameters are missing or the provided preprocessing/extraction methods are invalid.
    """
    if "preprocessing_dict" in kwargs:
        kwargs = kwargs["preprocessing_dict"]

    preprocessing = Preprocessing(kwargs["preprocessing"])
    mode = ExtractionMethod(kwargs["extract_method"])
    extraction = get_extraction(mode)(**kwargs)
    transforms = Transforms(extraction=extraction, **kwargs)
    caps_dir = kwargs["caps_directory"]
    data_tsv = kwargs["data_tsv"]
    return (
        get_preprocessing(preprocessing)(**kwargs),
        transforms,
        Path(caps_dir),
        Path(data_tsv),
    )
