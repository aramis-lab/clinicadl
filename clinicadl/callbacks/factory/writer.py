from abc import ABC
from typing import Any

from ..base import Callback
from ..training_state import _TrainingState


class _Writer(Callback):
    """Base class for callbacks."""

    def __init__(self):
        pass

    def on_train_begin(self, config: _TrainingState, **kwargs) -> None:
        config.maps._create_training_split(split=config.split)
        config.maps._create_summary_log()

        config.split.train_loader.dataset.write_json(
            config.maps.training.splits[config.split.index].caps_dataset_json
        )

    def on_train_end(self, config: _TrainingState, **kwargs) -> None:
        config.maps._add_lines_to_summary_log("=" * 15)

    def on_epoch_begin(self, config: _TrainingState, **kwargs) -> None:
        pass

    def on_epoch_end(self, config: _TrainingState, **kwargs) -> None:
        pass

    def on_batch_begin(self, config: _TrainingState, **kwargs) -> None:
        pass

    def on_batch_end(self, config: _TrainingState, **kwargs) -> None:
        pass

    def on_backward_begin(self, config: _TrainingState, **kwargs) -> None:
        pass

    def on_backward_end(self, config: _TrainingState, **kwargs) -> None:
        pass

    def on_validation_begin(self, config: _TrainingState, **kwargs) -> None:
        pass

    def on_validation_end(self, config: _TrainingState, **kwargs) -> None:
        pass
