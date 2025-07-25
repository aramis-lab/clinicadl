"""Callback to record training loss per batch and epoch."""
from typing import Any

import pandas as pd

from clinicadl.callbacks.training_state import _TrainingState
from clinicadl.dictionary.words import BATCH, EPOCH, LOSS

from .base import Callback


class _TrainingLoss(Callback):
    """
    Callback to record training loss per batch and epoch into a pandas DataFrame,
    and save it to a TSV file at the end of training.

    Attributes
    ----------
    df : pd.DataFrame
        DataFrame indexed by (epoch, batch) storing the training loss values.
    """

    def __init__(self):
        """
        Initialize the DataFrame to record training loss with MultiIndex (epoch, batch).
        """
        self.df = pd.DataFrame(columns=[EPOCH, BATCH, LOSS])
        self.df.set_index([EPOCH, BATCH], inplace=True)
        # self.df.at[(0, 0), LOSS] = 1.0

    def on_batch_end(self, config: _TrainingState, loss: float, **kwargs) -> None:
        """
        Called at the end of each batch to log the training loss.

        Parameters
        ----------
        config : _TrainingState
            Current training state.
        loss : float
            Loss value for the current batch.
        """
        self.df.at[(config.epoch, config.batch), LOSS] = loss

    def on_train_end(self, config: _TrainingState, **kwargs) -> None:
        """
        Called at the end of training to save the recorded losses to a TSV file.
        """
        training_tsv = config.maps.training.splits[config.split.index].logs.training_tsv
        training_tsv.parent.mkdir(parents=True, exist_ok=True)
        self.df.to_csv(training_tsv, sep="\t", index=True)
