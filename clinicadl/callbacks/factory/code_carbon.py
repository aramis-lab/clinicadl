from importlib.util import find_spec

from .base import Callback


def codecarbon_is_available() -> bool:
    return find_spec("codecarbon") is not None


class CodeCarbon(Callback):
    def __init__(self):
        if not codecarbon_is_available():
            raise ModuleNotFoundError(
                "`codecarbon` package must be installed. Run `pip install codecarbon`"
            )
        else:
            from codecarbon import EmissionsTracker

            self.tracker = EmissionsTracker(
                project_name="clinicadl", measure_power_secs=60
            )

    def on_train_begin(self, **kwargs):
        self.tracker.start()
        self.tracker.start_task("train")

    def on_train_end(self, **kwargs):
        self.tracker.stop_task("train")

    def on_epoch_begin(self, epoch: int, **kwargs):
        self.tracker.start_task("epoch")

    def on_epoch_end(self, epoch: int, **kwargs):
        self.tracker.stop_task("epoch")

    def on_batch_begin(self, batch: int, **kwargs):
        self.tracker.start_task("batch")

    def on_batch_end(self, batch: int, **kwargs):
        self.tracker.stop_task("batch")

    def on_backward_begin(self, **kwargs):
        self.tracker.start_task("backward")

    def on_backward_end(self, **kwargs):
        self.tracker.stop_task("backward")

    def on_validation_begin(self, **kwargs):
        self.tracker.start_task("validation")

    def on_validation_end(self, **kwargs):
        self.tracker.stop_task("validation")
