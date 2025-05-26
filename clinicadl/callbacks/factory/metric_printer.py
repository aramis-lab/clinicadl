import logging

import numpy as np

from .base import Callback


class MetricConsolePrinterCallback(Callback):
    """
    A :class:`TrainingCallback` printing the training logs in the console.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # make it print to the console.
        console = logging.StreamHandler()
        self.logger.addHandler(console)
        self.logger.setLevel(logging.INFO)

    def on_log(self, logs, **kwargs):
        logger = kwargs.pop("logger", self.logger)
        rank = kwargs.pop("rank", -1)

        if logger is not None and (rank == -1 or rank == 0):
            epoch_train_loss = logs.get("train_epoch_loss", None)
            epoch_eval_loss = logs.get("eval_epoch_loss", None)

            logger.info(
                "--------------------------------------------------------------------------"
            )
            if epoch_train_loss is not None:
                logger.info(f"Train loss: {np.round(epoch_train_loss, 4)}")
            if epoch_eval_loss is not None:
                logger.info(f"Eval loss: {np.round(epoch_eval_loss, 4)}")
            logger.info(
                "--------------------------------------------------------------------------"
            )
