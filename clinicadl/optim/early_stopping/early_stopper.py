import math

import numpy as np

from .config import EarlyStoppingConfig, Mode


class EarlyStopping(object):
    """
    To perform early stopping.

    Parameters
    ----------
    config : config
        The Early Stopping config object.
    """

    def __init__(self, config: EarlyStoppingConfig) -> None:
        self.mode = config.mode
        self.min_delta = config.min_delta
        self.patience = config.patience
        self.check_finite = config.check_finite
        self.upper_bound = config.upper_bound
        self.lower_bound = config.lower_bound
        self.is_better = self._get_comparison_function()
        self.reset()

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

    def _get_comparison_function(self):
        """Returns the comparison function."""
        if self.mode == Mode.MIN:
            return lambda value, best: value < best - self.min_delta
        if self.mode == Mode.MAX:
            return lambda value, best: value > best + self.min_delta
