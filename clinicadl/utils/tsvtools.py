from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from clinicadl.utils.dictionary.words import (
    DATASET_ID,
    N_SAMPLES,
    PARTICIPANT_ID,
    SESSION_ID,
)
from clinicadl.utils.exceptions import DataFrameError

from .dictionary.utils import SEP
from .typing import DataFrameType


def df_to_tsv(
    tsv_path: Path,
    df: pd.DataFrame,
    baseline: bool = False,
    drop_duplicates: bool = False,
) -> None:
    """
    Writes a :py:class:`pandas.Dataframe` into a ``TSV`` file.

    Parameters
    ----------
    tsv_path : Path
        The path to the ``TSV`` file.
    df : pd.DataFrame
        The :py:class:`pandas.Dataframe`.
    baseline : bool, default=False
        If ``True``, will save only the baseline session for each participant.
    drop_duplicates : bool, default=False
        If ``True``, it will keep only the first occurrence of a (participant, session) pair.
    """

    df = df.sort_values(by=[PARTICIPANT_ID, SESSION_ID])
    if baseline:
        df = df.drop_duplicates(subset=[PARTICIPANT_ID], keep="first")
    elif drop_duplicates:
        df = df.drop_duplicates(subset=[PARTICIPANT_ID, SESSION_ID], keep="first")
    df.to_csv(tsv_path, sep=SEP, index=False)


def read_data(
    data: DataFrameType,
    check_protected_names: bool = True,
    check_duplicates: bool = True,
) -> pd.DataFrame:
    """
    Reads an input :py:class:`pandas.Dataframe`, passed directly as a DataFrame or via a path, and
    performs checks on it.

    Parameters
    ----------
    data : DataFrameType
        The :py:class:`pandas.Dataframe` or a path to the DataFrame.
    check_protected_names : bool, default=True
        Whether to check if the DataFrame contains some column names that are protected.
    check_duplicates : bool, default=True
        Whether to check if the DataFrame contains duplicated (participant, session) pairs.

    Returns
    -------
    pd.DataFrame
        The dataframe, read and checked.

    Raises
    ------
    DataFrameError
        If the DataFrame is empty.
    DataFrameError
        If the columns ('participant_id', 'session_id') are not found in the DataFrame.
    DataFrameError
        If ``check_protected_names`` is ``True`` and the DataFrame contains columns named ``"n_samples"`` or ``"dataset_id"``.
    DataFrameError
        If ``check_duplicates`` is ``True`` and the DataFrame contains duplicated (participant, session) pairs.
    """
    if isinstance(data, (str, Path)):
        data = Path(data)
        data = pd.read_csv(data, sep=SEP)

    elif not isinstance(data, pd.DataFrame):
        raise TypeError(f"'data' must be a path or a DataFrame. Got: {data}")

    _check_df(data, check_protected_names, check_duplicates)

    return data


def _check_df(
    df: pd.DataFrame, check_protected_names: bool = True, check_duplicates: bool = True
) -> None:
    """
    Checks the input DataFrame.
    """
    if len(df) == 0:
        raise DataFrameError("The dataframe is empty!")

    if not {PARTICIPANT_ID, SESSION_ID}.issubset(set(df.columns.values)):
        raise DataFrameError(
            f"The dataframe is not in the correct format. "
            f"Columns should include {PARTICIPANT_ID, SESSION_ID}"
        )
    if check_protected_names:
        protected_names = {N_SAMPLES, DATASET_ID}
        if len(protected_names.intersection(set(df.columns.values))) > 0:
            raise DataFrameError(
                f"The dataframe contains some protected column names. "
                f"Please do not use names in {protected_names}"
            )

    if check_duplicates:
        duplicated_pairs = df[df[[PARTICIPANT_ID, SESSION_ID]].duplicated(keep=False)]
        if len(duplicated_pairs) > 0:
            raise DataFrameError(
                f"The dataframe contains duplicated (participant, session) pairs:\n"
                f"{duplicated_pairs}"
            )


def create_participants_sessions_df(
    participants_sessions: Iterable[tuple[str, str]],
) -> pd.DataFrame:
    """
    To create a :py:class:`pandas.Dataframe` with two columns named
    ``"participant_id"`` and ``"session_id"`` with the input participant
    and session ids.

    Parameters
    ----------
    participants_sessions : Iterable[tuple[str, str]]
        The (participant id, session id) couples.

    Returns
    -------
    pandas.DataFrame
        The output DataFrame.
    """
    return (
        pd.DataFrame(
            np.array(list(participants_sessions)),
            columns=[PARTICIPANT_ID, SESSION_ID],
        )
        .sort_values([PARTICIPANT_ID, SESSION_ID])
        .reset_index(drop=True)
    )
