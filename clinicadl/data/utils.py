# coding: utf8
from glob import glob
from logging import getLogger
from pathlib import Path
from typing import List, Tuple, Union

import pandas as pd

from clinicadl.data import preprocessing
from clinicadl.transforms import extraction
from clinicadl.transforms.transforms import Transforms
from clinicadl.utils.enum import ExtractionMethod, PreprocessingMethod
from clinicadl.utils.exceptions import ClinicaDLTSVError
from clinicadl.utils.iotools.utils import read_preprocessing

logger = getLogger("clinicadl.data.utils")

PARTICIPANT_ID = "participant_id"
SESSION_ID = "session_id"


def df_to_tsv(
    name: str, results_path: Path, df: pd.DataFrame, baseline: bool = False
) -> None:
    """
    Write Dataframe into a TSV file and drop duplicates

    Parameters
    ----------
    name: str
        Name of the tsv file
    results_path: str (path)
        Path to the folder
    df: DataFrame
        DataFrame you want to write in a TSV file.
        Columns must include ["participant_id", "session_id"].
    baseline: bool
        If True, there is only baseline session for each subject.
    """

    df.sort_values(by=["participant_id", "session_id"], inplace=True)
    if baseline:
        df.drop_duplicates(subset=["participant_id"], keep="first", inplace=True)
    else:
        df.drop_duplicates(
            subset=["participant_id", "session_id"], keep="first", inplace=True
        )
    # df = df[["participant_id", "session_id"]]
    df.to_csv(results_path / name, sep="\t", index=False)


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
    if extract_method == ExtractionMethod.SLICE:
        return extraction.Slice
    elif extract_method == ExtractionMethod.IMAGE:
        return extraction.Image
    elif extract_method == ExtractionMethod.PATCH:
        return extraction.Patch
    else:
        raise ValueError(f"Preprocessing {extract_method.value} is not implemented.")


def get_preprocessing(
    preprocessing_type: Union[str, PreprocessingMethod],
) -> type[preprocessing.Preprocessing]:
    """
    Retrieves the preprocessing class based on the specified preprocessing type.

    Args:
        preprocessing_type (Union[str, PreprocessingMethod]): The preprocessing type as either a string or a `Preprocessing` enum.

    Returns:
        type[preprocessing.Preprocessing]: The corresponding preprocessing configuration class.

    Raises:
        ValueError: If the provided `preprocessing_type` is not supported or is invalid.
    """
    preprocessing_type = PreprocessingMethod(preprocessing_type)
    if preprocessing_type == PreprocessingMethod.T1_LINEAR:
        return preprocessing.PreprocessingT1
    elif preprocessing_type == PreprocessingMethod.PET_LINEAR:
        return preprocessing.PreprocessingPET
    elif preprocessing_type == PreprocessingMethod.FLAIR_LINEAR:
        return preprocessing.PreprocessingFlair
    elif preprocessing_type == PreprocessingMethod.CUSTOM:
        return preprocessing.PreprocessingCustom
    elif preprocessing_type == PreprocessingMethod.DWI_DTI:
        return preprocessing.PreprocessingDTI
    else:
        raise ValueError(
            f"Preprocessing {preprocessing_type.value} is not implemented."
        )


def get_infos_from_json(
    json_path: Path,
) -> Tuple[preprocessing.Preprocessing, Transforms, Path, Path]:
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
) -> Tuple[preprocessing.Preprocessing, Transforms, Path, Path]:
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

    preprocessing = PreprocessingMethod(kwargs["preprocessing"])
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


class Mask:
    """To handle masks in ClinicaDL. More precisely, it makes the difference
    between a mask passed as a file name, that corresponds to a common mask,
    and a mask passed as a suffix (a simple string), that corresponds to a mask
    specific to each subject.

    For example, `Mask("masks/mask.nii.gz")` will be understood has a common
    mask, where as `Mask("mask")` will be understood has a specific mask.

    In the latter case, it is expected that all the (subject, session) studied
    have the associated mask in their CAPS folders. It will look for files with
    the suffix `mask` in these folders.

    Parameters
    ----------
    filename : Union[str, Path]
        the mask, passed as a path or a suffix.
    """

    def __init__(self, mask: Union[str, Path]) -> None:
        if isinstance(mask, Path):
            if not self._check_path(mask):
                raise ValueError(
                    f"The mask has been passed as a Path object (got {mask}), but no such file exists."
                )
            self.common_mask = True
            self.mask = Path(mask)

        elif isinstance(mask, str):
            if self._check_path(mask):
                self.common_mask = True
                self.mask = Path(mask)
            else:
                self.common_mask = False
                self.mask = mask

    @staticmethod
    def _check_path(mask_path: Union[str, Path]) -> bool:
        """Checks if the mask file exists."""
        mask_path = Path(mask_path)
        return mask_path.is_file()

    def get_associated_mask(self, filename: Union[str, Path]) -> Path:
        """
        Returns the mask associated to an image.

        If the mask is common to all subjects and sessions, the method will
        simply return it. On the other hand, if the mask is specific to each
        (subject, session), the method will use the input `filename` to get
        the associated mask.

        Parameters
        ----------
        filename : Union[str, Path]
            the image whose associated mask is to be found.

        Returns
        -------
        Path :
            the path to the mask associated to the image.

        Raises
        ------
        ValueError
            if the associated mask doesn't exist.

        Examples
        --------
        >>> mask=Mask("seg")
        >>> mask.get_associated_mask("sub-001_ses-M000_T1w.nii.gz")
        PosixPath('sub-001_ses-M000_seg.nii.gz')

        >>> mask=Mask("masks/leftHippocampus.nii.gz")
        >>> mask.get_associated_mask("sub-001_ses-M000_T1w.nii.gz")
        PosixPath('masks/leftHippocampus.nii.gz')
        """

        if self.common_mask:
            return self.mask
        else:
            filename = Path(filename)
            without_extension = str(filename).rstrip("".join(filename.suffixes))
            suffix = without_extension.rsplit("_", maxsplit=1)[-1]
            mask_file = str(filename).replace(f"_{suffix}.", f"_{self.mask}.")
            if not self._check_path(mask_file):
                raise ValueError(
                    f"A mask associated to {str(filename)} was expected "
                    f"to be found in {mask_file}, but there is no such file."
                )

            return Path(mask_file)
