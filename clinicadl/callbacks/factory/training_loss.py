import pandas as pd

from clinicadl.dictionary.words import BATCH, EPOCH, LOSS
from clinicadl.utils.config.training import _TrainingState

from .base import Callback


class TrainingLoss(Callback):
    """TO COMPLETE"""

    def __init__(self):
        """
        Initialize the dataframe to record training loss and time.

        Returns
        -------
        pd.DataFrame
            A dataframe to log loss and computation time per epoch and batch.
        """
        self.df = pd.DataFrame(columns=[EPOCH, BATCH, LOSS])
        self.df.set_index([EPOCH, BATCH], inplace=True)
        self.df.at[(0, 0), LOSS] = 1.0

    def on_train_end(self, config: _TrainingState, **kwargs):
        training_tsv = config.maps.splits[config.split.index].logs.training_tsv
        (training_tsv.parent).mkdir(parents=True, exist_ok=True)
        self.df.to_csv(training_tsv, sep="\t", index=True)

    def on_batch_end(self, config: _TrainingState, loss: float, **kwargs):
        self.df.at[(config.epoch, config.batch), LOSS] = loss
