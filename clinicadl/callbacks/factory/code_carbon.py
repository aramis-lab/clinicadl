from importlib.util import find_spec
from pathlib import Path

from clinicadl.trainer.config import _TrainingConfig

from .base import Callback


def codecarbon_is_available() -> bool:
    return find_spec("codecarbon") is not None


class CodeCarbon(Callback):
    def __init__(self, maps_path: Path):
        if not codecarbon_is_available():
            raise ModuleNotFoundError(
                "`codecarbon` package must be installed. Run `pip install codecarbon`"
            )
        else:
            from codecarbon import EmissionsTracker

            codecarbon_dir = maps_path / "codecarbon"

            if not codecarbon_dir.exists():
                codecarbon_dir.mkdir(parents=True, exist_ok=True)

            self.tracker = EmissionsTracker(
                project_name="clinicadl",
                output_dir=str(codecarbon_dir),
            )

    def on_train_begin(self, config: _TrainingConfig, **kwargs):
        self.tracker.start()
        self.tracker.start_task("train")

    def on_train_end(self, config: _TrainingConfig, **kwargs):
        self.tracker.stop_task("train")

    def on_epoch_begin(self, config: _TrainingConfig, **kwargs):
        self.tracker.start_task("epoch")

    def on_epoch_end(self, config: _TrainingConfig, **kwargs):
        self.tracker.stop_task("epoch")

    def on_batch_begin(self, config: _TrainingConfig, **kwargs):
        self.tracker.start_task("batch")

    def on_batch_end(self, config: _TrainingConfig, **kwargs):
        self.tracker.stop_task("batch")

    def on_backward_begin(self, config: _TrainingConfig, **kwargs):
        self.tracker.start_task("backward")

    def on_backward_end(self, config: _TrainingConfig, **kwargs):
        self.tracker.stop_task("backward")

    def on_validation_begin(self, config: _TrainingConfig, **kwargs):
        self.tracker.start_task("validation")

    def on_validation_end(self, config: _TrainingConfig, **kwargs):
        self.tracker.stop_task("validation")
