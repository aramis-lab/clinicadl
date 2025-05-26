import importlib

import numpy as np

from .base import Callback


def comet_is_available():
    return importlib.util.find_spec("comet_ml") is not None


class Comet(Callback):  # pragma: no cover
    """
    A :class:`TrainingCallback` integrating the experiment tracking tool
    `comet_ml` (https://www.comet.com/site/).

    It allows users to store their configs, monitor
    their trainings and compare runs through a graphic interface. To be able use this feature
    you will need:

    - the package `comet_ml` installed in your virtual env. If not you can install it with

    .. code-block::

        $ pip install comet_ml
    """

    def __init__(self):
        if not comet_is_available():
            raise ModuleNotFoundError(
                "`comet_ml` package must be installed. Run `pip install comet_ml`"
            )

        else:
            import comet_ml

            self._comet_ml = comet_ml

    def setup(
        self,
        api_key: str = None,
        project_name: str = "pythae_experiment",
        workspace: str = None,
        offline_run: bool = False,
        offline_directory: str = "./",
        **kwargs,
    ):
        """
        Setup the CometCallback.

        args:
            training_config (BaseTraineronfig): The training configuration used in the run.

            model_config (BaseAEConfig): The model configuration used in the run.

            api_key (str): Your personal comet-ml `api_key`.

            project_name (str): The name of the wandb project to use.

            workspace (str): The name of your comet-ml workspace

            offline_run: (bool): Whether to run comet-ml in offline mode.

            offline_directory (str): The path to store the offline runs. They can to be
                synchronized then by running `comet upload`.
        """

        self.is_initialized = True

        if not offline_run:
            experiment = self._comet_ml.Experiment(
                api_key=api_key, project_name=project_name, workspace=workspace
            )
            experiment.log_other("Created from", "pythae")
        else:
            experiment = self._comet_ml.OfflineExperiment(
                api_key=api_key,
                project_name=project_name,
                workspace=workspace,
                offline_directory=offline_directory,
            )
            experiment.log_other("Created from", "pythae")

        experiment.log_parameters({}, prefix="training_config/")
        experiment.log_parameters({}, prefix="model_config/")

    def on_train_begin(self, **kwargs):
        model_config = kwargs.pop("model_config", None)
        if not self.is_initialized:
            self.setup(training_config, model_config=model_config)

    def on_log(self, logs, **kwargs):
        global_step = kwargs.pop("global_step", None)

        experiment = self._comet_ml.get_global_experiment()
        experiment.log_metrics(logs, step=global_step, epoch=global_step)

    def on_prediction_step(self, **kwargs):
        global_step = kwargs.pop("global_step", None)

        column_names = ["images_id", "truth", "reconstruction", "normal_generation"]

        true_data = kwargs.pop("true_data", None)
        reconstructions = kwargs.pop("reconstructions", None)
        generations = kwargs.pop("generations", None)

        experiment = self._comet_ml.get_global_experiment()

        if (
            true_data is not None
            and reconstructions is not None
            and generations is not None
        ):
            for i in range(len(true_data)):
                experiment.log_image(
                    np.moveaxis(true_data[i].cpu().detach().numpy(), 0, -1),
                    name=f"{i}_truth",
                    step=global_step,
                )
                experiment.log_image(
                    np.clip(
                        np.moveaxis(reconstructions[i].cpu().detach().numpy(), 0, -1),
                        0,
                        255.0,
                    ),
                    name=f"{i}_reconstruction",
                    step=global_step,
                )
                experiment.log_image(
                    np.clip(
                        np.moveaxis(generations[i].cpu().detach().numpy(), 0, -1),
                        0,
                        255.0,
                    ),
                    name=f"{i}_normal_generation",
                    step=global_step,
                )

    def on_train_end(self, **kwargs):
        experiment = self._comet_ml.config.get_global_experiment()
        experiment.end()
