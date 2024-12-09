from pathlib import Path
from typing import List, Optional

import pandas as pd
from pydantic import BaseModel, PositiveInt


class SubjectsSessionsSplit(BaseModel):
    """
    Dataclass to store train and validation splits for subjects and sessions.
    """

    train: pd.DataFrame
    validation: pd.DataFrame


class KFoldConfig(BaseModel):
    """
    Configuration for K-Fold cross-validation splits.
    """

    split_dir: Optional[Path] = None
    subset_name: str = "validation"
    n_splits: PositiveInt = 5
    stratification: Optional[List[str]] = None
    ignore_demographics: bool = False
    valid_longitudinal: bool = False
