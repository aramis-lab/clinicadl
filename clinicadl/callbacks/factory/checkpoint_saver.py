import torch

from clinicadl.callbacks.training_state import _TrainingState
from clinicadl.dictionary.suffixes import PTH, TAR
from clinicadl.dictionary.words import CHECKPOINT, EPOCH, MODEL, OPTIMIZER

from .base import Callback


class _CheckpointSaver(Callback):
    """
    Callback that saves the current state of the model and optimizer at the end of each epoch.

    This callback ensures that the training progress is preserved by saving the model weights,
    the optimizer state and current epoch.

    These files are stored in `.pt.tar` format, in the maps, in a temporary directory associated
    with the current training split.

    Notes
    -----
    .. note:
        - This callback is added automatically at the beginning of the training.
        - Used internally for restoring the latest state when training is resumed.

    """

    def on_train_begin(self, config: _TrainingState, **kwargs) -> None:
        """
        Check that training and validation DataLoaders are initialized.

        This method ensures that data loading has been configured correctly before training begins.
        """
        if config.split.train_loader is None:
            raise ValueError(
                "The split has no train_loader defined. Please run `get_dataloader()`"
            )
        if config.split.val_loader is None:
            raise ValueError(
                "The split has no val_loader defined. Please run `get_dataloader()`"
            )

    def on_epoch_end(self, config: _TrainingState, **kwargs) -> None:
        """
        Save the current model and optimizer state at the end of each epoch.

        This includes the epoch number and corresponding state dicts for both the
        model and optimizer. These are saved in the `tmp` directory of the current split in the maps.
        """
        model_weights = {
            MODEL: config.model.network.state_dict(),
            EPOCH: config.epoch,
        }
        checkpoint_path = config.maps.splits[config.split.index].tmp.path / (
            MODEL + PTH + TAR
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

    def on_train_end(self, config: _TrainingState, **kwargs) -> None:
        """
        Remove the temporary storage used for the latest checkpoint after training completes.
        """
        config.maps.splits[config.split.index].tmp.remove()
