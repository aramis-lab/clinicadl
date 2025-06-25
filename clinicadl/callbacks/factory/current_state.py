import torch

from clinicadl.dictionary.suffixes import PTH, TAR
from clinicadl.dictionary.words import CHECKPOINT, EPOCH, MODEL, OPTIMIZER
from clinicadl.utils.config.training import _TrainingState

from .base import Callback


class CurrentState(Callback):
    """Base class for callbacks."""

    def __init__(self):
        "test"
        pass

    def on_train_begin(self, config: _TrainingState, **kwargs):
        if config.split.train_loader is None:
            raise ValueError(
                "The split has no train_loader defined. Please run `get_dataloader()`"
            )
        if config.split.val_loader is None:
            raise ValueError(
                "The split has no val_loader defined. Please run `get_dataloader()`"
            )

    def on_train_end(self, config: _TrainingState, **kwargs):
        pass

    def on_epoch_begin(self, config: _TrainingState, **kwargs):
        pass

    def on_epoch_end(self, config: _TrainingState, **kwargs):
        model_weights = {
            MODEL: config.model.network.state_dict(),
            EPOCH: config.epoch,
        }
        checkpoint_path = config.maps.splits[config.split.index].tmp.path / (
            CHECKPOINT + PTH + TAR
        )
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(model_weights, checkpoint_path)

        optim_weights = {
            MODEL: config.model.optimizer.state_dict(),
            EPOCH: config.epoch,
        }
        optim_path = config.maps.splits[config.split.index].tmp.path / (
            OPTIMIZER + PTH + TAR
        )

        torch.save(optim_weights, optim_path)

    def on_batch_begin(self, config: _TrainingState, **kwargs):
        pass

    def on_batch_end(self, config: _TrainingState, **kwargs):
        pass

    def on_backward_begin(self, config: _TrainingState, **kwargs):
        pass

    def on_backward_end(self, config: _TrainingState, **kwargs):
        pass

    def on_validation_begin(self, config: _TrainingState, **kwargs):
        pass

    def on_validation_end(self, config: _TrainingState, **kwargs):
        pass
