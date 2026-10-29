import random
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from clinicadl.utils.dictionary.words import (
    AGE,
    DIAGNOSIS,
    PARTICIPANT_ID,
    SCORE,
    SESSION_ID,
    SEX,
)
from clinicadl.utils.exceptions import DataFrameError

from .dictionary.utils import TSV_SEP
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
    df.to_csv(tsv_path, sep=TSV_SEP, index=False)


def read_df(
    data: DataFrameType,
    protected_names: Optional[Iterable[str]] = None,
    check_duplicates: bool = True,
) -> pd.DataFrame:
    """
    Reads an input :py:class:`pandas.Dataframe`, passed directly as a DataFrame or via a path, and
    performs checks on it.

    Parameters
    ----------
    data : DataFrameType
        The :py:class:`pandas.Dataframe` or a path to the DataFrame.
    protected_names : Optional[Iterable[str]], default=None
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
        If participant or session ids are not strings starting with ``"sub-"``
        or ``"ses-"``, respectively.
    DataFrameError
        If the DataFrame contains columns in``protected_names``.
    DataFrameError
        If ``check_duplicates`` is ``True`` and the DataFrame contains duplicated (participant, session) pairs.
    """
    if isinstance(data, (str, Path)):
        data = Path(data)
        data = pd.read_csv(data, sep=TSV_SEP)

    elif not isinstance(data, pd.DataFrame):
        raise TypeError(f"'data' must be a path or a DataFrame. Got: {data}")

    _check_df(data, protected_names, check_duplicates)

    return data


def _check_df(
    df: pd.DataFrame,
    protected_names: Optional[Iterable[str]],
    check_duplicates: bool = True,
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

    for column, prefix in ((PARTICIPANT_ID, "sub-"), (SESSION_ID, "ses-")):
        valid_ids = df[column].map(
            lambda value: isinstance(value, str) and value.startswith(prefix)
        )
        if not valid_ids.all():
            raise DataFrameError(
                f"The '{column}' column must contain strings starting with '{prefix}'."
            )

    if protected_names:
        if len(set(protected_names).intersection(set(df.columns.values))) > 0:
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


def generate_dataframe(
    n_participants: int,
    n_sessions: int = 1,
    exact_n_sessions: bool = True,
    score_decline_per_year: tuple[float, float] = (0.01, 0.1),
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Generates a random but realistic DataFrame.

    The DataFrame contains the following columns:

    - ``participant_id``: (**str**) the participant ids;
    - ``session_id``: (**str**) the session ids;
    - ``age``: (**int**) the age of the participants;
    - ``sex``: (**str**, ``"M"`` or ``"F"``) the sex of the participants;
    - ``diagnosis``: (**str**, ``"CN"``, ``"AD"`` or ``"MCI"``) the cognitive state of the participants;
    - ``score``: (**float**) a cognitive score (between 0 and 1) of the participants.

    ``diagnosis`` is determined by ``score``:

    - ``CN``: ``score`` > 0.6
    - ``MCI``: 0.4 < ``score`` <= 0.6
    - ``AD``: ``score`` <= 0.4

    Parameters
    ----------
    n_participants : int
        The number of participants to generate.
    n_sessions : int, default=1
        The maximum number of sessions per participant.
    exact_n_sessions : bool, default=True
        If ``True``, each participant will have exactly ``n_sessions``. Otherwise, the
        number of sessions for each participant will be randomly sampled between ``1`` and
        ``n_sessions``.
    score_decline_per_year : tuple[float, float], default=(0.01, 0.1)
        Percentage decline in the score each year. The actual decline will be randomly sampled
        in this interval.
    seed : Optional[int], default=None
        A seed to control the randomness of the generation.

    Returns
    -------
    pd.DataFrame
        The DataFrame generated.

    Examples
    --------
    .. code-block:: python

        >>> from clinicadl.utils.tsvtools import generate_dataframe
        >>> generate_dataframe(3, 4, exact_n_sessions=False, score_decline_per_year=(0.1, 0.5), seed=8)
        participant_id session_id age sex diagnosis score
        0   sub-1      ses-M000   84  F   MCI       0.42618520092964296
        1   sub-1      ses-M004   84  F   AD        0.3982946664806757
        2   sub-2      ses-M000   78  M   AD        0.2603286896781217
        3   sub-2      ses-M008   78  M   AD        0.213712327652985
        4   sub-2      ses-M016   79  M   AD        0.1637583203276657
        5   sub-3      ses-M000   49  M   CN        0.9599623029969878
        6   sub-3      ses-M007   49  M   CN        0.7100904475751515
        7   sub-3      ses-M009   49  M   CN        0.6556287812937336
        8   sub-3      ses-M017   50  M   MCI       0.512899542646471

    """
    rng = random.Random(seed)
    session_counts = [
        rng.randint(n_sessions if exact_n_sessions else 1, n_sessions)
        for _ in range(n_participants)
    ]

    diagnoses = ["CN", "MCI", "AD"] * ((n_participants + 2) // 3)
    diagnoses = diagnoses[:n_participants]
    rng.shuffle(diagnoses)

    sexes = ["M", "F"] * ((n_participants + 1) // 2)
    sexes = sexes[:n_participants]
    rng.shuffle(sexes)

    participant_width = max(1, len(str(n_participants)))
    score_range = {"CN": (0.61, 1.0), "MCI": (0.41, 0.6), "AD": (0.01, 0.4)}
    records = []
    for participant, (n_sub_sessions, diagnosis, sex) in enumerate(
        zip(session_counts, diagnoses, sexes), start=1
    ):
        score = rng.uniform(*score_range[diagnosis])
        session_months = [0]
        for _ in range(1, n_sub_sessions):
            session_months.append(session_months[-1] + rng.randint(1, 12))
        first_age = rng.randint(20, 90 - session_months[-1] // 12)

        previous_month = 0
        for month in session_months:
            if month > previous_month:
                annual_retention = 1 - rng.uniform(*score_decline_per_year)
                score *= annual_retention ** ((month - previous_month) / 12)
                diagnosis = "CN" if score > 0.6 else "MCI" if score > 0.4 else "AD"

            records.append(
                {
                    PARTICIPANT_ID: f"sub-{participant:0{participant_width}d}",
                    SESSION_ID: f"ses-M{month:03d}",
                    AGE: first_age + month // 12,
                    SEX: sex,
                    DIAGNOSIS: diagnosis,
                    SCORE: score,
                }
            )
            previous_month = month

    return pd.DataFrame(
        records, columns=[PARTICIPANT_ID, SESSION_ID, AGE, SEX, DIAGNOSIS, SCORE]
    )
