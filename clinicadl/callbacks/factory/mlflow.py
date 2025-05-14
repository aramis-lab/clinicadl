from importlib.util import find_spec

from .base import Callback


def mlflow_is_available() -> bool:
    return find_spec("mlflow") is not None


class MLFLOWCallback(Callback):
    def __init__(self):
        if not mlflow_is_available():
            raise ModuleNotFoundError(
                "`mlflow` package must be installed. Run `pip install mlflow`"
            )
        else:
            import mlflow

            self._mlflow = mlflow

    def on_train_begin(self, **kwargs):
        # OLD CODE FOR MLFLOW
        # try:
        #     experiment_id = self._mlflow.create_experiment(
        #         f"clinicadl-{maps_name}",
        #         artifact_location=Path.cwd().joinpath("mlruns").as_uri(),
        #     )

        # except mlflow.exceptions.MlflowException:
        #     self._mlflow.set_experiment(maps_name)

        # self._mlflow.start_run(experiment_id=experiment_id, run_name=f"split-{split}")
        # self._mlflow.autolog()
        # config_bis = copy(config)
        # for cle, valeur in config.items():
        #     if cle == "preprocessing_dict":
        #         del config_bis[cle]
        # config = config_bis
        # self._mlflow.log_params(config)

        pass

    def on_train_end(self, **kwargs):
        pass
