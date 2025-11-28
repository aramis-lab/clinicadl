from copy import deepcopy
from typing import Optional

import pandas as pd


def subset_df(
    df: pd.DataFrame, participants_sessions: Optional[list[tuple[str, str]]] = None
) -> pd.DataFrame:
    if not participants_sessions:
        return deepcopy(df)
    data = df.set_index(["participant_id", "session_id"])
    data = data.loc[participants_sessions]

    return data.reset_index()
