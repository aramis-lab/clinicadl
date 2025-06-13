from tqdm import tqdm

from clinicadl.utils.config.training import _TrainingState

from .base import Callback


class ProgressBarCallback(Callback):
    """
    A :class:`TrainingCallback` printing the training progress bar.
    """

    def __init__(self):
        self.train_progress_bar = None
        self.eval_progress_bar = None

    def on_train_begin(self, config: _TrainingState, **kwargs):
        """TO COMPLETE"""
        epoch = config.epoch
        train_loader = config.split.train_loader
        rank = kwargs.pop("rank", -1)
        if train_loader is not None and (rank == 0 or rank == -1):
            self.train_progress_bar = tqdm(
                total=len(train_loader),
                unit="batch",
                desc=f"Training of epoch {epoch}/{config.optim.epochs}",
            )

    def on_batch_begin(self, config: _TrainingState, **kwargs):
        """TO COMPLETE"""
        epoch = config.epoch
        val_loader = config.split.val_loader
        rank = kwargs.pop("rank", -1)
        if val_loader is not None and (rank == 0 or rank == -1):
            self.eval_progress_bar = tqdm(
                total=len(val_loader),
                unit="batch",
                desc=f"Eval of epoch {epoch}/{config.optim.epochs}",
            )

    def on_batch_end(self, config: _TrainingState, **kwargs):
        if self.train_progress_bar is not None:
            self.train_progress_bar.update(1)

    def on_validation_end(self, config: _TrainingState, **kwargs):
        if self.eval_progress_bar is not None:
            self.eval_progress_bar.update(1)

    def on_epoch_end(self, config: _TrainingState, **kwags):
        if self.train_progress_bar is not None:
            self.train_progress_bar.close()

        if self.eval_progress_bar is not None:
            self.eval_progress_bar.close()
