import json
from pathlib import Path
from typing import Generator, List, Optional, Sequence, Tuple, Union

import pandas as pd
from pydantic import NonNegativeInt, PositiveInt
from sklearn.model_selection import StratifiedKFold

from clinicadl.dataset.datasets.caps_dataset import CapsDataset
from clinicadl.dataset.utils import tsv_to_df
from clinicadl.splitter.split import Split
from clinicadl.splitter.splitter.kfold import KFold, KFoldConfig
from clinicadl.splitter.splitter.splitter import SubjectsSessionsSplit
from clinicadl.tsvtools.tsvtools_utils import extract_baseline, retrieve_longitudinal
from clinicadl.utils.exceptions import ClinicaDLTSVError


def _write_to_csv(df: pd.DataFrame, file_path: Path) -> None:
    """
    Save DataFrame to a TSV file.

    Parameters
    ----------
    df : pd.DataFrame
        Data to save.
    file_path : Path
        Destination file path.

    """
    if file_path.is_file():
        raise FileExistsError(f"File {file_path} already exists.")
    df.reset_index(drop=True, inplace=True)
    df.to_csv(file_path, sep="\t", index=False)


def _check_stratification(
    df: pd.DataFrame,
    ignore_demographics: bool,
    stratification: Optional[List[str]] = None,
) -> Optional[List[str]]:
    """
    Checks and validates the specified stratification columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    ignore_demographics : bool
        If True, ignore demographic columns for balancing.
    stratification : List[str], optional
        List of columns to stratify on.

    Returns
    -------
    List[str], optional
        Validated list of stratification columns or None if no stratification is applied.

    Raises
    ------
    ValueError
        If specified stratification columns are missing or if stratification conflicts with demographic handling.
    ClinicaDLTSVError
        If required demographic columns ('age', 'sex') are missing when not ignored.
    """

    if stratification:
        missing_columns = set(stratification) - set(df.columns)
        if missing_columns:
            raise ValueError(
                f"Stratification variables {missing_columns} not found in dataset."
            )
        if ignore_demographics:
            raise ValueError("Cannot stratify while ignoring demographics.")

        if not {"age", "sex"}.issubset(df.columns):
            raise ClinicaDLTSVError(
                "Dataset missing 'age' or 'sex' columns for demographic balancing."
            )
        # TODO: check if we want to always stratify on age and sex

    elif not ignore_demographics:
        stratification = ["age", "sex"]
    return stratification


def preprocess_stratification(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    ignore_demographics: bool = False,
) -> List[str]:
    """
    Preprocess stratification columns by creating labels for each subject.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    columns : Optional[List[str]]
        Columns to stratify on.
    ignore_demographics : bool
        If True, ignore demographic columns.

    Returns
    -------
    List[str]
        List of stratification labels for the dataset.
    """
    columns = _check_stratification(df, ignore_demographics, columns)
    if not columns:
        return ["0"] * len(df)

    labels = []
    for col in columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            # Numerical column: bin into 5 equal groups or fewer if unique values < 5
            labels.append(
                pd.cut(df[col], bins=min(5, df[col].nunique()), labels=False).astype(
                    str
                )
            )
        else:
            labels.append(df[col].astype(str))
    return ["_".join(label) for label in zip(*labels)]
