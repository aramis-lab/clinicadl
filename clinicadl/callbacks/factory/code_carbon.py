# TODO : Not working at the moment

from importlib.util import find_spec

from clinicadl.utils.config.training import _TrainingState

from .base import Callback


class CodeCarbon(Callback):
    """
    CodeCarbon callback to estimate and track carbon emissions from your computer, quantify and analyze their impact.
    See https://codecarbon.io/ for more information.
    """

    def __init__(self):
        if not self.is_available():
            raise ModuleNotFoundError(
                "`codecarbon` package must be installed. Run `pip install codecarbon`"
            )

    @staticmethod
    def is_available() -> bool:
        """Check if codecarbon package is installed and available"""
        return find_spec("codecarbon") is not None

    def set_tracker(self, config: _TrainingState):
        """Set the tracker

        Parameters
        ----------
        config : _TrainingState
            The training config
        """

        from codecarbon import (
            EmissionsTracker,  # pylint: disable=import-outside-toplevel
        )

        codecarbon_dir = config.maps.path / "codecarbon"

        if not codecarbon_dir.exists():
            codecarbon_dir.mkdir(parents=True, exist_ok=True)

        self.tracker = EmissionsTracker(  # pylint: disable=attribute-defined-outside-init
            project_name="clinicadl",
            output_dir=str(codecarbon_dir),
        )

    def on_train_begin(self, config: _TrainingState, **kwargs):
        self.set_tracker(config)
        self.tracker.start()
        self.tracker.start_task("train")

    def on_train_end(self, config: _TrainingState, **kwargs):
        self.tracker.stop_task("train")

    def on_epoch_begin(self, config: _TrainingState, **kwargs):
        self.tracker.start_task("epoch")

    def on_epoch_end(self, config: _TrainingState, **kwargs):
        self.tracker.stop_task("epoch")

    def on_batch_begin(self, config: _TrainingState, **kwargs):
        self.tracker.start_task("batch")

    def on_batch_end(self, config: _TrainingState, **kwargs):
        self.tracker.stop_task("batch")

    def on_backward_begin(self, config: _TrainingState, **kwargs):
        self.tracker.start_task("backward")

    def on_backward_end(self, config: _TrainingState, **kwargs):
        self.tracker.stop_task("backward")

    def on_validation_begin(self, config: _TrainingState, **kwargs):
        self.tracker.start_task("validation")

    def on_validation_end(self, config: _TrainingState, **kwargs):
        self.tracker.stop_task("validation")
