from __future__ import annotations

from logging import getLogger
from typing import Any, Mapping

import numpy as np

from clinicadl.metrics.enum import Optimum

logger = getLogger("clinicadl.callbacks.implemented.utils.QuantityMonitoring")


class QuantityMonitoring:
    """
    Utility class to monitor a quantity and determine
    at each step if the quantity was improved.
    """

    def __init__(
        self,
        name: str,
        min_delta: float,
        mode: Optimum,
    ):
        self.name = name
        self.min_delta = min_delta
        self.mode = Optimum(mode)
        self.reset()

    def reset(self) -> None:
        """Resets the best value and counter."""
        if self.mode == Optimum.MIN:
            self.best = np.inf
        elif self.mode == Optimum.MAX:
            self.best = -np.inf

        self.num_non_improvements = 0

    def step(self, value: float, log: bool = False) -> bool:
        """
        Determines if the new value improves the quantity,
        and increments or resets the count accordingly.

        Parameters
        ----------
        value : float
            The new value of the monitored quantity.
        log : bool, default=True
            Whether to log when no improvement.

        Returns
        -------
        bool
            If the quantity has been improved.
        """
        if self._is_better(value):
            self.num_non_improvements = 0
            self.best = value

            return True

        self.num_non_improvements += 1
        if log:
            logger.debug(
                "No improvement in '%s' for %s evaluation step(s).",
                self.name,
                self.num_non_improvements,
            )

        return False

    def state_dict(self) -> Mapping[str, Any]:
        """
        Returns the state of the monitoring.

        Returns
        -------
        Mapping[str, Any]
            The state of the monitoring in a dictionary.
        """
        return {"best": self.best, "num_non_improvements": self.num_non_improvements}

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """
        Resets the monitoring to a given state.

        Parameters
        ----------
        state_dict : Mapping[str, Any]
            The desired state of the monitoring, as returned by :py:meth:`state_dict`.
        """
        self.best = state_dict["best"]
        self.num_non_improvements = state_dict["num_non_improvements"]

    def _is_better(self, value: float) -> bool:
        """Does the new value improves the monitored quantity?"""
        if self.mode == Optimum.MIN:
            return value < self.best - self.min_delta
        elif self.mode == Optimum.MAX:
            return value > self.best + self.min_delta
