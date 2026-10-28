import re
from pathlib import Path

import pandas as pd
import pytest

from clinicadl.utils.exceptions import DataFrameError
from clinicadl.utils.tsvtools import (
    create_participants_sessions_df,
    df_to_tsv,
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
