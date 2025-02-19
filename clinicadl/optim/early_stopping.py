from __future__ import annotations

import math
from enum import Enum
from typing import Callable, Optional

import numpy as np
from pydantic import (
    NonNegativeFloat,
    PositiveInt,
    computed_field,
    model_validator,
)

from clinicadl.utils.config import ClinicaDLConfig


class Mode(str, Enum):
    """Supported mode for Early Stopping."""

    MIN = "min"
    MAX = "max"


class EarlyStopping(ClinicaDLConfig):
    """
    To perform early stopping.

    Parameters
    ----------
    config : config
        The Early Stopping config object.
    """

    patience: Optional[PositiveInt] = None
    min_delta: NonNegativeFloat = 0.0
    mode: Mode = Mode.MIN
    check_finite: bool = True
    upper_bound: Optional[float] = None
    lower_bound: Optional[float] = None
    best: float = 0.0
    num_bad_epochs: int = 0

    @model_validator(mode="after")
    def init_reset(self):
        print(self)
        if self.upper_bound is not None and self.lower_bound is not None:
            if self.upper_bound < self.lower_bound:
                raise ValueError(
                    "The upper bound must be greater than the lower bound."
                )

        if self.best == 0:
            self.reset()

        return self

    @computed_field
    @property
    def is_better(self) -> Callable:
        return self._get_comparison_function()

    def reset(self) -> None:
        """Resets the epoch count and the best value."""
        if self.mode == Mode.MIN:
            self.best = np.inf
        if self.mode == Mode.MAX:
            self.best = -np.inf
        self.num_bad_epochs = 0

    def step(self, value: float) -> bool:
        """
        Decides whether to stop the training or not, depending
        on the value of the last epoch.

        Parameters
        ----------
        value : float
            The value obtained during the last epoch.

        Returns
        -------
        bool
            The decision.
        """
        if self.check_finite and (math.isinf(value) or math.isnan(value)):
            return True

        if self.upper_bound is not None and (value > self.upper_bound):
            return True

        if self.lower_bound is not None and (value < self.lower_bound):
            return True

        if self.patience is None:
            return False

        if self.is_better(value, self.best):
            self.num_bad_epochs = 0
            self.best = value
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _get_comparison_function(self) -> Callable:
        """Returns the comparison function."""
        if self.mode == Mode.MIN:
            return lambda value, best: value < best - self.min_delta
        if self.mode == Mode.MAX:
            return lambda value, best: value > best + self.min_delta

        raise ValueError(f"Invalid mode: {self.mode}")
