from copy import deepcopy
from typing import Optional

import pandas as pd
import torch
import torchio as tio

from clinicadl.data.datasets import Dataset, MultiSamplesDataset
from clinicadl.data.datatypes import T1Linear
from clinicadl.data.structures import Sample


def subset_df(
    df: pd.DataFrame, participants_sessions: Optional[list[tuple[str, str]]] = None
) -> pd.DataFrame:
    if not participants_sessions:
        return deepcopy(df)
    data = df.set_index(["participant_id", "session_id"])
    data = data.loc[participants_sessions]

    return data.reset_index()


class CustomMultiSamplesDataset(MultiSamplesDataset):
    def __init__(self, df):
        self._df = df
        self.evaluation = False

    def train(self):
        self.evaluation = False

    def eval(self):
        self.evaluation = True

    def __getitem__(self, idx) -> Sample:
        return Sample(
            participant=self.get_sample_info(idx, "participant_id"),
            session=self.get_sample_info(idx, "session_id"),
            image=tio.ScalarImage(tensor=torch.ones(1, 1, 1, 1)),
            datatype=T1Linear(),
            image_path=str(idx),
        )


class CustomDataset(Dataset):
    def __init__(self, df):
        self._df = df
        self.evaluation = False

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    def train(self):
        self.evaluation = False

    def eval(self):
        self.evaluation = True

    def subset(self, particpants_sessions):
        if isinstance(particpants_sessions, pd.DataFrame):
            particpants_sessions = list(
                zip(
                    particpants_sessions["participant_id"],
                    particpants_sessions["session_id"],
                )
            )

        sub_df = (
            self.df.set_index(["participant_id", "session_id"])
            .loc[particpants_sessions]
            .reset_index()
        )
        return type(self)(sub_df)

    def get_sample_info(self, idx, column):
        return self.df.iloc[idx].at[column]

    def get_participant_session_couples(self):
        return set(zip(self._df["participant_id"], self._df["session_id"]))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx) -> Sample:
        return Sample(
            participant=self.get_sample_info(idx, "participant_id"),
            session=self.get_sample_info(idx, "session_id"),
            image=tio.ScalarImage(tensor=torch.ones(1, 1, 1, 1)),
            datatype=T1Linear(),
            image_path=str(idx),
        )
