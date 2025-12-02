from copy import deepcopy
from typing import Optional

import pandas as pd
import torch
import torchio as tio

from clinicadl.data.datasets.multi_samples import MultiSamplesDataset
from clinicadl.data.datasets.output import Sample
from clinicadl.data.datatypes import T1Linear


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
