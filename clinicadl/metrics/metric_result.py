from logging import getLogger
from typing import Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, field_validator

logger = getLogger("clinicadl.metric")


class MetricResult(BaseModel):
    name: Tuple[str, ...] = ()
    value: Tuple[float, ...] = ()
    lower_ci: Tuple[float, ...] = ()
    upper_ci: Tuple[float, ...] = ()
    se: Tuple[float, ...] = ()

    @field_validator("*", mode="before")
    def validator_diagnoses(cls, v):
        """Transforms a list to a tuple."""
        if isinstance(v, list):
            return tuple(v)
        return v

    def append(
        self,
        name: str,
        value: float,
        lower_ci: float = np.nan,
        upper_ci: float = np.nan,
        se: float = np.nan,
    ):
        self.name += (name,)
        self.value += (value,)
        self.lower_ci += (lower_ci,)
        self.upper_ci += (upper_ci,)
        self.se += (se,)

    def get_value(self, name_: str) -> float:
        idx = self.name.index(name_)
        return self.value[idx]

    def to_df(self) -> pd.DataFrame:
        out = pd.DataFrame(
            columns=["Metrics", "Values", "Lower bound CI", "Upper bound CI", "SE"]
        )

        out["Metrics"] = list(self.name)
        out["Values"] = list(self.value)
        out["Lower bound CI"] = list(self.lower_ci)
        out["Upper bound CI"] = list(self.upper_ci)
        out["SE"] = list(self.se)

        return out.T
