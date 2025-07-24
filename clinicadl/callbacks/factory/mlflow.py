# TODO : Not working at the moment


from importlib.util import find_spec
from typing import Union

import pandas as pd

from clinicadl.callbacks.training_state import _TrainingState
from clinicadl.dictionary.suffixes import PTH, TAR
from clinicadl.dictionary.words import MODEL, OPTIMIZER

from .base import Callback


class MLflow(Callback):
    """
    A training callback that integrates with the experiment tracking tool `MLflow <https://mlflow.org/>`_.

    This callback enables logging of training configurations, metrics, and artifacts,
    allowing users to monitor experiments and compare training runs through the MLflow
    graphical interface.

    Requirements
    ------------
    - The `mlflow` package must be installed in your Python environment.
      You can install it with:

    .. code-block:: bash

        pip install mlflow

    .. note::
        - MLflow supports local file logging, remote tracking servers, and integration
          with various cloud platforms.
        - This callback is useful for reproducibility and large-scale experiment tracking.

    Examples
    --------
    .. code-block:: python

        from clinicadl.callbacks import MLflow

        mlflow_callback = MLflow()
        handler = _CallbacksHandler(callbacks=[mlflow_callback])


    .. seealso::
        - `MLflow documentation <https://mlflow.org/>`_.
    """

    def __init__(
        self, experiment_name: str = "default", run_name: Union[str, None] = None
    ):
        if not self.is_available():
            raise ModuleNotFoundError(
                "`mlflow` package must be installed. Run `pip install mlflow`"
            )

        else:
            import mlflow

            self._mlflow = mlflow

        self.experiment_name = experiment_name
        self.run_name = run_name
        self.run = None

    @staticmethod
    def is_available() -> bool:
        """TO COMPLETE"""
        return find_spec("mlflow") is not None

    def on_train_begin(self, config: _TrainingState, **kwargs) -> None:
        """Initialize MLflow experiment and start a new run."""
        self._mlflow.set_experiment(self.experiment_name)
        self.run = self._mlflow.start_run(run_name=self.run_name)

        # Optional: Log hyperparameters (example)
        if hasattr(config, "model") and hasattr(config.model, OPTIMIZER):
            for param_group in config.model.optimizer.param_groups:
                for k, v in param_group.items():
                    if isinstance(v, (int, float, str)):
                        self._mlflow.log_param(k, v)

    def on_epoch_end(self, config: _TrainingState, **kwargs) -> None:
        """
        Log all metrics from the current epoch to MLflow.

        Assumes `config.metrics.df` is a pandas DataFrame with epochs as the index.
        """
        if config.metrics.df is not None and not config.metrics.df.empty:
            epoch_metrics = config.metrics.df.loc[config.epoch]
            if epoch_metrics is not None:
                for metric_name, value in epoch_metrics.items():
                    if value is not None and not pd.isna(value):
                        self._mlflow.log_metric(
                            metric_name, float(value), step=config.epoch
                        )

    def on_train_end(self, config: _TrainingState, **kwargs) -> None:
        """
        Log the final model checkpoint and optimizer state as MLflow artifacts,
        then end the MLflow run.
        """
        tmp_path = config.maps.training.splits[config.split.index].tmp.path
        model_file = tmp_path / MODEL + PTH + TAR
        optimizer_file = tmp_path / OPTIMIZER + PTH + TAR

        # Log model checkpoint
        if model_file.exists():
            self._mlflow.log_artifact(str(model_file), artifact_path=MODEL)

        # Log optimizer checkpoint
        if optimizer_file.exists():
            self._mlflow.log_artifact(str(optimizer_file), artifact_path=OPTIMIZER)

        self._mlflow.end_run()
