from importlib.util import find_spec

from .base import Callback


def mlflow_is_available() -> bool:
    return find_spec("mlflow") is not None


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
        if not mlflow_is_available():
            raise ModuleNotFoundError(
                "`mlflow` package must be installed. Run `pip install mlflow`"
            )

        else:
            import mlflow

            self._mlflow = mlflow

    def setup(
        self,
        run_name: str = None,
        **kwargs,
    ):
        """
        Setup the MLflowCallback.

        args:
            training_config (BaseTrainerConfig): The training configuration used in the run.

            model_config (BaseAEConfig): The model configuration used in the run.

            run_name (str): The name to apply to the current run.
        """
        self.is_initialized = True
        self._mlflow.start_run(run_name=run_name)

        logger.info(
            f"MLflow run started with run_id={self._mlflow.active_run().info.run_id}"
        )
        self._mlflow.log_params({})

    def on_train_begin(self, **kwargs):
        model_config = kwargs.pop("model_config", None)
        if not self.is_initialized:
            self.setup(training_config, model_config=model_config)

    def on_log(self, logs, **kwargs):
        global_step = kwargs.pop("global_step", None)

        logs = rename_logs(logs)
        metrics = {}
        for k, v in logs.items():
            if isinstance(v, (int, float)):
                metrics[k] = v

        self._mlflow.log_metrics(metrics=metrics, step=global_step)

    def on_train_end(self, **kwargs):
        self._mlflow.end_run()

    def __del__(self):
        # if the previous run is not terminated correctly, the fluent API will
        # not let you start a new run before the previous one is killed
        if (
            callable(getattr(self._mlflow, "active_run", None))
            and self._mlflow.active_run() is not None
        ):
            self._mlflow.end_run()
