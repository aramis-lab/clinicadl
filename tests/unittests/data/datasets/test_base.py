import re

import pandas as pd
import pytest

from clinicadl.data.datasets import Dataset


class MyDataset(Dataset):
    def __init__(self):
        self._df = pd.DataFrame(
            {
                "participant_id": ["sub-000", "sub-000", "sub-000", "sub-001"],
                "session_id": ["ses-M000", "ses-M000", "ses-M001", "ses-M002"],
            }
        )

    def train(self):
        pass

    def eval(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass

    def get_sample_info(self, idx, column):
        pass


class TestDataset:
    def test_get_participant_session_couples(self):
        dataset = MyDataset()
        assert dataset.get_participant_session_couples() == set(
            [
                ("sub-000", "ses-M000"),
                ("sub-000", "ses-M001"),
                ("sub-001", "ses-M002"),
            ]
        )

    @pytest.mark.parametrize(
        "dataframe",
        [
            [("sub-000", "ses-M000"), ("sub-001", "ses-M002")],
            pd.DataFrame(
                {
                    "participant_id": ["sub-000", "sub-100", "sub-001"],
                    "session_id": ["ses-M000", "ses-M100", "ses-M002"],
                }
            ),
        ],
    )
    def test_subset(self, dataframe):
        dataset = MyDataset()
        subset = dataset.subset(dataframe)
        assert isinstance(subset, MyDataset)
        pd.testing.assert_frame_equal(
            subset.df,
            pd.DataFrame(
                {
                    "participant_id": ["sub-000", "sub-000", "sub-001"],
                    "session_id": ["ses-M000", "ses-M000", "ses-M002"],
                }
            ),
        )

    def test_subset_2(self, tmp_path):
        dataset = MyDataset()
        df = pd.DataFrame(
            {
                "participant_id": ["sub-000", "sub-001"],
                "session_id": ["ses-M000", "ses-M002"],
            }
        )
        df.to_csv(tmp_path / "df.tsv", sep="\t", index=False)
        subset = dataset.subset(tmp_path / "df.tsv")
        pd.testing.assert_frame_equal(
            subset.df,
            pd.DataFrame(
                {
                    "participant_id": ["sub-000", "sub-000", "sub-001"],
                    "session_id": ["ses-M000", "ses-M000", "ses-M002"],
                }
            ),
        )

    def test_subset_3(self):
        dataset = MyDataset()
        with pytest.raises(
            RuntimeError,
            match=re.escape(
                "No (participant, session) pairs are in the dataset. This would lead to an empty dataset!"
            ),
        ):
            dataset.subset([("sub-100", "ses-M000")])
