import re
from pathlib import Path

import pandas as pd
import pytest
from pandas.api.types import is_float_dtype, is_integer_dtype, is_string_dtype

from clinicadl.utils.exceptions import DataFrameError
from clinicadl.utils.tsvtools import (
    create_participants_sessions_df,
    df_to_tsv,
    generate_dataframe,
    read_df,
)


def test_create_participants_sessions_df():
    df = create_participants_sessions_df(
        [("sub-000", "ses-M000"), ("sub-100", "ses-M003")]
    )
    pd.testing.assert_frame_equal(
        df,
        pd.DataFrame(
            {
                "participant_id": ["sub-000", "sub-100"],
                "session_id": ["ses-M000", "ses-M003"],
            }
        ),
    )


BIDS = Path(__file__).parents[1] / "resources" / "bids"


class TestReadData:
    @pytest.mark.parametrize("from_path", [False, True])
    @pytest.mark.parametrize(
        "column,prefix", [("participant_id", "sub-"), ("session_id", "ses-")]
    )
    @pytest.mark.parametrize("invalid_id", ["000", "", "other-000", None, 1])
    def test_invalid_id(self, tmp_path, from_path, column, prefix, invalid_id):
        df = pd.DataFrame(
            {
                "participant_id": ["sub-000", "sub-100"],
                "session_id": ["ses-M000", "ses-M003"],
            }
        )
        df.loc[1, column] = invalid_id
        data = df
        if from_path:
            data = tmp_path / "participants.tsv"
            df.to_csv(data, sep="\t", index=False)

        with pytest.raises(
            DataFrameError,
            match=re.escape(
                f"The '{column}' column must contain strings starting with '{prefix}'."
            ),
        ):
            read_df(data, check_duplicates=False)

    @pytest.mark.parametrize("from_path", [False, True])
    def test_valid_ids_unchanged(self, tmp_path, from_path):
        df = pd.DataFrame(
            {
                "participant_id": ["sub-001", "sub-ADNI123"],
                "session_id": ["ses-M000", "ses-20261021"],
                "age": [60, 70],
            }
        )
        expected = df.copy(deep=True)
        data = df
        if from_path:
            data = tmp_path / "participants.tsv"
            df.to_csv(data, sep="\t", index=False)

        pd.testing.assert_frame_equal(read_df(data), expected)
        pd.testing.assert_frame_equal(df, expected)

    def test_path(self):
        df = read_df(BIDS / "participantsXsessions.tsv")
        assert len(df) == 8

    def test_empty(self):
        with pytest.raises(DataFrameError, match="The dataframe is empty!"):
            read_df(pd.DataFrame())

    def test_protected_names(self):
        with pytest.raises(
            DataFrameError,
            match=re.escape(
                "The dataframe contains some protected column names. Please do not use names in ['abc']"
            ),
        ):
            read_df(
                pd.DataFrame(
                    {
                        "participant_id": ["sub-000"],
                        "session_id": ["ses-M000"],
                        "abc": ["x"],
                    }
                ),
                protected_names=["abc"],
            )

    def test_check_duplicated(self):
        with pytest.raises(
            DataFrameError,
            match=r"The dataframe contains duplicated \(participant, session\) pairs:.*",
        ):
            read_df(
                pd.DataFrame(
                    {
                        "participant_id": ["sub-000", "sub-000"],
                        "session_id": ["ses-M000", "ses-M000"],
                    }
                ),
            )
        read_df(
            pd.DataFrame(
                {
                    "participant_id": ["sub-000", "sub-000"],
                    "session_id": ["ses-M000", "ses-M000"],
                }
            ),
            check_duplicates=False,
        )

    @pytest.mark.parametrize(
        "df",
        [
            pd.DataFrame(
                {
                    "session_id": ["ses-M000", "ses-M000"],
                }
            ),
            pd.DataFrame(
                {
                    "participant_id": ["sub-000", "sub-000"],
                }
            ),
        ],
    )
    def test_participant_session(self, df):
        with pytest.raises(
            DataFrameError,
            match=re.escape(
                "The dataframe is not in the correct format. Columns should include ('participant_id', 'session_id')"
            ),
        ):
            read_df(df)


class TestDfToTsv:
    def test(self, tmp_path):
        df = pd.DataFrame(
            {
                "participant_id": ["sub-010", "sub-000", "sub-010", "sub-010"],
                "session_id": ["ses-M003", "ses-M000", "ses-M000", "ses-M000"],
            }
        )
        df_to_tsv(tmp_path / "df.tsv", df)
        pd.testing.assert_frame_equal(
            pd.read_csv(tmp_path / "df.tsv", sep="\t"),
            pd.DataFrame(
                {
                    "participant_id": ["sub-000", "sub-010", "sub-010", "sub-010"],
                    "session_id": ["ses-M000", "ses-M000", "ses-M000", "ses-M003"],
                }
            ),
        )

    def test_baseline(self, tmp_path):
        df = pd.DataFrame(
            {
                "participant_id": ["sub-010", "sub-000", "sub-010"],
                "session_id": ["ses-M003", "ses-M000", "ses-M000"],
            }
        )
        df_to_tsv(tmp_path / "df.tsv", df, baseline=True)
        pd.testing.assert_frame_equal(
            pd.read_csv(tmp_path / "df.tsv", sep="\t"),
            pd.DataFrame(
                {
                    "participant_id": ["sub-000", "sub-010"],
                    "session_id": ["ses-M000", "ses-M000"],
                }
            ),
        )

    def test_duplicates(self, tmp_path):
        df = pd.DataFrame(
            {
                "participant_id": ["sub-010", "sub-000", "sub-010", "sub-010"],
                "session_id": ["ses-M003", "ses-M000", "ses-M000", "ses-M000"],
            }
        )
        df_to_tsv(tmp_path / "df.tsv", df, drop_duplicates=True)
        pd.testing.assert_frame_equal(
            pd.read_csv(tmp_path / "df.tsv", sep="\t"),
            pd.DataFrame(
                {
                    "participant_id": ["sub-000", "sub-010", "sub-010"],
                    "session_id": ["ses-M000", "ses-M000", "ses-M003"],
                }
            ),
        )


@pytest.fixture
def df() -> pd.DataFrame:
    return generate_dataframe(20, 10)


class TestGenerateDataframe:
    @pytest.mark.parametrize(
        "n_participants,n_sessions,exact_n_sessions,seed,expected_n_sessions",
        [(5, 3, True, None, [3] * 5), (5, 3, False, 0, [2, 2, 1, 2, 3])],
    )
    def test_size(
        self, n_participants, n_sessions, exact_n_sessions, seed, expected_n_sessions
    ):
        df = generate_dataframe(
            n_participants, n_sessions, exact_n_sessions=exact_n_sessions, seed=seed
        )
        assert (
            df.groupby(["participant_id"]).count()["session_id"].tolist()
            == expected_n_sessions
        )

    def test_max_size(self):
        df = generate_dataframe(100, 3, exact_n_sessions=False)
        assert (df.groupby("participant_id")["session_id"].apply(len) <= 3).all()

    @pytest.mark.parametrize(
        "column", ["participant_id", "session_id", "age", "sex", "score", "diagnosis"]
    )
    def test_non_nans(self, column, df):
        assert df[column].count() == 200

    @pytest.mark.parametrize(
        "column", ["participant_id", "session_id", "sex", "diagnosis"]
    )
    def test_str_columns(self, column, df):
        assert is_string_dtype(df[column])

    def test_first_session(self, df):
        assert (
            df.groupby("participant_id")
            .head(1)["session_id"]
            .apply(lambda x: int(x.replace("ses-M", "")))
            == 0
        ).all()

    def test_age(self, df):
        assert is_integer_dtype(df["age"])
        assert ((20 <= df["age"]) & (df["age"] <= 90)).all()

    def test_sex(self, df):
        assert set(df["sex"].unique()) == {"M", "F"}

    def test_score(self, df):
        assert is_float_dtype(df["score"])
        assert ((0 <= df["score"]) & (df["score"] <= 1)).all()

    def test_diagnosis(self, df):
        assert set(df["diagnosis"].unique()) == {"CN", "MCI", "AD"}

    def test_score_diagnosis(self, df):
        assert (
            (0.4 < df[df["diagnosis"] == "MCI"]["score"])
            & (df[df["diagnosis"] == "MCI"]["score"] <= 0.6)
        ).all()
        assert (df[df["diagnosis"] == "AD"]["score"] <= 0.4).all()
        assert (0.6 < df[df["diagnosis"] == "CN"]["score"]).all()

    @pytest.mark.parametrize(
        "column, prefix", [("participant_id", "sub-"), ("session_id", "ses-")]
    )
    def test_id_n_digits(self, column, prefix, df):
        """Test that all the participant/session ids have the same number of digits."""
        ids_len = df[column].apply(lambda x: len(x.replace(prefix, "")))
        assert len(ids_len.unique()) == 1

    def test_age_consistency(self, df):
        """Test that, for every participant, the age is increasing when the session id is increasing."""
        assert (
            df.sort_values(["participant_id", "session_id"])
            .groupby("participant_id")["age"]
            .is_monotonic_increasing.all()
        )

    def test_sex_consistency(self, df):
        """Test that, for every participant, the age is increasing when the session id is increasing."""
        assert (
            df.groupby("participant_id")["sex"].apply(lambda x: len(x.unique()) == 1)
        ).all()

    def test_diagnosis_evolution(self, df):
        """Test that the diagnosis is not always constant."""
        assert not (
            df.groupby("participant_id")["diagnosis"].apply(
                lambda x: len(x.unique()) == 1
            )
        ).all()

    def test_scores_evolution(self, df):
        """Test that the score is not constant, even with same diagnosis."""
        assert not (
            df.groupby(["participant_id", "diagnosis"])["score"].apply(
                lambda x: len(x.unique()) == 1
            )
        ).all()

    def test_ad_stays_ad(self, df):
        """Test that AD diagnosis does not evolve."""
        first_session_diagnosis = (
            df.groupby("participant_id")
            .apply(lambda x: x.sort_values("session_id").head(1)["diagnosis"])
            .droplevel(1)
        )
        ad_from_beginning = first_session_diagnosis[
            first_session_diagnosis == "AD"
        ].index
        assert (
            df[df["participant_id"].isin(ad_from_beginning)]["diagnosis"] == "AD"
        ).all()
