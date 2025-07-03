from pathlib import Path
from typing import Optional

import pandas as pd

from clinicadl.dictionary.suffixes import TSV
from clinicadl.dictionary.words import BASELINE, DATASET_ID, PARTICIPANT_ID, SESSION_ID
from clinicadl.tsvtools.utils import read_data
from clinicadl.utils.typing import DataType

__all__ = [
    "read_and_format_data",
    "extract_baseline",
    "write_to_tsv",
    "find_available_split_dir",
]


def read_and_format_data(data: DataType) -> pd.DataFrame:
    """
    Reads the input dataframe, passed directly as a dataframe or via a path, performs
    checks on it, and formats it in a uniform way.

    Parameters
    ----------
    data : Union[str, Path, pd.DataFrame]
        The DataFrame, as a pandas DataFrame or a path.

    Returns
    -------
    pd.DataFrame
        The dataframe, read, checked and formatted.

    Raises
    ------
    ValueError
        If 'data' is not a str, a Path or a pandas DataFrame.
    ClinicaDLTSVError
        If the DataFrame is empty.
    ClinicaDLTSVError
        If the required columns ('participant_id', 'session_id') are not found in the DataFrame.
    """
    if (
        isinstance(data, pd.DataFrame) and DATASET_ID in data.columns.names
    ):  # for unpaired datasets
        data = (
            data.stack(DATASET_ID).reset_index(level=DATASET_ID).reset_index(drop=True)
        )

    df = read_data(data, check_duplicates=False, check_protected_names=False)

    return df


def extract_baseline(
    df: pd.DataFrame, columns: Optional[list[str]] = None
) -> pd.DataFrame:
    """
    From a dataframe, returns the baseline dataframe, i.e. the dataframe with
    only the first session for each subject.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe, with at least two columns "participant_id" and "session_id".
    columns : Optional[list[str]], (optional, default=None)
        Which column to include in the output dataframe. As ``df`` can contain several times
        the same (participant_id, baseline_session) (e.g. for ConcatDataset), extract_baseline will
        check that there is indeed a unique value per column for each (participant_id, baseline_session).

    Returns
    -------
    pd.DataFrame
        The baseline dataframe, with the wanted columns.

    Raises
    ------
    ValueError
        If a column contain more than one unique value for a (participant_id, baseline_session).
    """

    first_sessions = (
        df.sort_values(SESSION_ID)
        .drop_duplicates(PARTICIPANT_ID)[[PARTICIPANT_ID, SESSION_ID]]
        .set_index(PARTICIPANT_ID)[SESSION_ID]
    )

    baseline = first_sessions.to_frame(SESSION_ID).reset_index()

    if columns is None:
        return baseline
    else:
        columns = [PARTICIPANT_ID, SESSION_ID] + columns

    baseline = baseline.merge(df[columns], how="left", on=[PARTICIPANT_ID, SESSION_ID])

    group_cols = [PARTICIPANT_ID, SESSION_ID]
    grouped = baseline.groupby(group_cols)

    resolved_rows = []

    for (pid, sid), group in grouped:
        for col in group.columns:
            values = group[col].dropna().unique()
            if len(values) > 1:
                raise ValueError(
                    f"More than one value found in the dataframe for ({pid}, {sid}) in '{col}': {list(values)}. "
                    "ClinicaDL can't decide which value to take."
                )
        resolved_rows.append(group.iloc[0])

    baseline_cleaned = pd.DataFrame(resolved_rows).reset_index(drop=True)

    return baseline_cleaned


def write_to_tsv(
    baseline_df: pd.DataFrame,
    split_dir: Path,
    subset_name: str,
    all_df: Optional[pd.DataFrame] = None,
    longitudinal: bool = True,
) -> None:
    """
    Save baseline and longitudinal splits of a DataFrame in TSV files.

    Parameters
    ----------
    baseline_df : pd.DataFrame
        DataFrame containing the subset (e.g., train/test/validation) to save.
    split_dir : Path
        Directory where the TSV files will be saved.
    subset_name : str
        Name of the subset (e.g., "train", "test", etc.) used in the output filenames.
    all_df : Optional[pd.DataFrame], (optional, default=None)
        Full dataset including all sessions, used to retrieve longitudinal data.
    longitudinal : bool, (optional, default=True)
        Whether to save the longitudinal data subset.

    Raises
    ------
    FileExistsError
        If any of the output files already exist in the specified directory.
    ValueError
        If `longitudinal` is True but `all_df` is None.
    """
    # Save the baseline data
    baseline_file = (split_dir / f"{subset_name}_{BASELINE}").with_suffix(TSV)
    _write_to_tsv(baseline_df, baseline_file)

    if longitudinal:
        if all_df is None:
            raise ValueError(
                "The full dataframe (`all_df`) must be provided to save longitudinal data."
            )

        # Retrieve longitudinal data and save it
        longitudinal_file = (split_dir / f"{subset_name}").with_suffix(TSV)
        long_df = _retrieve_longitudinal(all_df, baseline_df)
        _write_to_tsv(long_df, longitudinal_file)


def _write_to_tsv(df: pd.DataFrame, file_path: Path) -> None:
    """
    Save a DataFrame to a TSV file, ensuring the file does not already exist.
    """
    if file_path.exists():
        raise FileExistsError(
            f"File {file_path} already exists. Operation aborted to prevent overwriting."
        )

    df.to_csv(file_path, sep="\t", index=False)


def _retrieve_longitudinal(
    all_df: pd.DataFrame, baselin_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Gets a longitudinal subset of 'all_df' from the subjects in 'baseline_df'.
    """
    longitudinal_df = (
        all_df.set_index([PARTICIPANT_ID]).loc[baselin_df[PARTICIPANT_ID]].reset_index()
    )
    return (
        longitudinal_df[baselin_df.columns]
        .drop_duplicates()
        .sort_values([PARTICIPANT_ID, SESSION_ID])
        .reset_index(drop=True)
    )


def find_available_split_dir(source_dir: Path, split_name: str) -> Path:
    """
    Finds an available subdirectory in 'source_dir' name for a split
    named 'split_name'.

    Parameters
    ----------
    source_dir : Path
        The source directory, where the split directory will be.
    split_name : str
        The name of the split.

    Returns
    -------
    Path
        The complete path to the new split directory.
    """
    split_number = 1
    folder_name = split_name
    while (source_dir / folder_name).is_dir():
        split_number += 1
        folder_name = f"{split_name}_{split_number}"

    return source_dir / folder_name
