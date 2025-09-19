# Note: This callback is currently working but in a very basic way. It could be improved in the future.
# For example, by allowing more parameters to be set (like tracking more specific hardware, etc.)
# See https://codecarbon.io/ for more details.

from importlib.util import find_spec
from logging import getLogger

from clinicadl.callbacks.training_state import _TrainingState

from .base import Callback

CODECARBON = "codecarbon"


class CodeCarbon(Callback):
    """
    `CodeCarbon <https://codecarbon.io/>`_ callback to estimate and track carbon emissions from your computer, quantify and analyze their impact.
    See https://codecarbon.io/ for more information.
    """

    def __init__(self):
        if not self.is_available():
            raise ModuleNotFoundError(
                "`codecarbon` must be installed. Run: pip install codecarbon"
            )

    @staticmethod
    def is_available() -> bool:
        """Check if codecarbon package is installed and available"""
        return find_spec(CODECARBON) is not None

    def set_tracker(self, config: _TrainingState):
        """
        Initialize the CodeCarbon tracker.
        """
        from codecarbon import EmissionsTracker, OfflineEmissionsTracker
        from codecarbon.output_methods.logger import LoggerOutput

        codecarbon_dir = (
            config.maps.training.splits[config.split.index].path / CODECARBON
        )
        codecarbon_dir.mkdir(parents=True, exist_ok=True)

        try:
            self.tracker = EmissionsTracker(
                project_name="clinicadl",
                output_dir=str(codecarbon_dir),
                save_to_logger=True,
                logging_logger=LoggerOutput(getLogger("clinicadl.codecarbon")),
            )
        except Exception:
            # fallback if EmissionsTracker fails (e.g. environment not detected)
            self.tracker = OfflineEmissionsTracker(
                project_name="clinicadl",
                output_dir=str(codecarbon_dir),
            )

    def on_train_begin(self, config: _TrainingState, **kwargs) -> None:
        self.set_tracker(config)
        self.tracker.start()

    def on_train_end(self, config: _TrainingState, **kwargs) -> None:
        self.tracker.stop()
