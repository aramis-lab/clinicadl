# TODO : Not working at the moment


from importlib.util import find_spec
from typing import Optional

from clinicadl.utils.config.training import _TrainingState

from .base import Callback


class MLFlow(Callback):  # pragma: no cover
    """
    A :class:`TrainingCallback` integrating the experiment tracking tool
    `mlflow` (https://mlflow.org/).

    It allows users to store their configs, monitor their trainings
    and compare runs through a graphic interface. To be able use this feature you will need:

        - the package `mlfow` installed in your virtual env. If not you can install it with

        .. code-block::

            $ pip install mlflow
    """

    def __init__(self):
        if not self.is_available():
            raise ModuleNotFoundError(
                "`mlflow` package must be installed. Run `pip install mlflow`"
            )

        else:
            import mlflow

            self._mlflow = mlflow

    @staticmethod
    def is_available() -> bool:
        """TO COMPLETE"""
        return find_spec("mlflow") is not None

    def setup(
        self,
        run_name: Optional[str] = None,
        **kwargs,
    ):
        """
        Setup the MLflowCallback.

        """
        self.is_initialized = True
        self._mlflow.start_run(run_name=run_name)

        self._mlflow.log_params({})

    def on_train_begin(self, config: _TrainingState, **kwargs):
        if not self.is_initialized:
            self.setup(run_name=config.maps.path.name)

    def on_train_end(self, config: _TrainingState, **kwargs):
        self._mlflow.end_run()

    def __del__(self):
        # if the previous run is not terminated correctly, the fluent API will
        # not let you start a new run before the previous one is killed
        if (
            callable(getattr(self._mlflow, "active_run", None))
            and self._mlflow.active_run() is not None
        ):
            self._mlflow.end_run()
