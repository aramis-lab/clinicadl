from tqdm import tqdm

from clinicadl.utils.config.training import _TrainingConfig

from .base import Callback


class ProgressBarCallback(Callback):
    """
    A :class:`TrainingCallback` printing the training progress bar.
    """

    def __init__(self):
        self.train_progress_bar = None
        self.eval_progress_bar = None

    def on_train_step_begin(self, config: _TrainingConfig, **kwargs):
        """TO COMPLETE"""
        epoch = kwargs.pop("epoch", None)
        train_loader = kwargs.pop("train_loader", None)
        rank = kwargs.pop("rank", -1)
        if train_loader is not None and (rank == 0 or rank == -1):
            self.train_progress_bar = tqdm(
                total=len(train_loader),
                unit="batch",
                desc=f"Training of epoch {epoch}/{config.optim.epochs}",
            )

    def on_batch_begin(self, config: _TrainingConfig, **kwargs):
        """TO COMPLETE"""
        epoch = kwargs.pop("epoch", None)
        eval_loader = kwargs.pop("eval_loader", None)
        rank = kwargs.pop("rank", -1)
        if eval_loader is not None and (rank == 0 or rank == -1):
            self.eval_progress_bar = tqdm(
                total=len(eval_loader),
                unit="batch",
                desc=f"Eval of epoch {epoch}/{config.optim.epochs}",
            )

    def on_batch_end(self, config: _TrainingConfig, **kwargs):
        if self.train_progress_bar is not None:
            self.train_progress_bar.update(1)

    def on_validation_end(self, config: _TrainingConfig, **kwargs):
        if self.eval_progress_bar is not None:
            self.eval_progress_bar.update(1)

    def on_epoch_end(self, config: _TrainingConfig, **kwags):
        if self.train_progress_bar is not None:
            self.train_progress_bar.close()

        if self.eval_progress_bar is not None:
            self.eval_progress_bar.close()
