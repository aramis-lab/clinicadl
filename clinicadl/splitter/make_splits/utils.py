from pathlib import Path
from typing import Optional

import pandas as pd

from clinicadl.tsvtools.utils import retrieve_longitudinal


def _write_to_csv(df: pd.DataFrame, file_path: Path) -> None:
    """
    Save a DataFrame to a TSV file, ensuring the file does not already exist.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to save.
    file_path : Path
        Path to the destination TSV file.

    Raises
    ------
    FileExistsError
        If the file already exists at the specified path.
    """
    if file_path.exists():
        raise FileExistsError(
            f"File {file_path} already exists. Operation aborted to prevent overwriting."
        )

    # Reset index for consistency and save as a TSV file
    df.reset_index(drop=True, inplace=True)
    df.to_csv(file_path, sep="\t", index=False)


def write_to_csv(
    df: pd.DataFrame,
    split_dir: Path,
    all_df: Optional[pd.DataFrame] = None,
    subset_name: str = "train",
    longitudinal: bool = True,
) -> None:
    """
    Save baseline and longitudinal splits of a DataFrame to TSV files.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the subset (e.g., train/test/validation) to save.
    split_dir : Path
        Directory where the TSV files will be saved.
    all_df : Optional[pd.DataFrame], optional
        Full dataset including all sessions, used to retrieve longitudinal data.
    subset_name : str, default="train"
        Name of the subset (e.g., "train", "test", etc.) used in the output filenames.
    longitudinal : bool, default=True
        Whether to generate and save the longitudinal data subset.

    Raises
    ------
    FileExistsError
        If any of the output files already exist in the specified directory.
    ValueError
        If `longitudinal` is True but `all_df` is None, as longitudinal data cannot be generated.
    """
    # Save the baseline data
    baseline_file = split_dir / f"{subset_name}_baseline.tsv"
    _write_to_csv(df, baseline_file)

    if longitudinal:
        if all_df is None:
            raise ValueError(
                "The full dataset (`all_df`) must be provided to generate longitudinal data."
            )

        # Retrieve longitudinal data and save it
        longitudinal_file = split_dir / f"{subset_name}.tsv"
        long_df = retrieve_longitudinal(df, all_df)
        _write_to_csv(long_df, longitudinal_file)
