from importlib.util import find_spec

import numpy as np

from clinicadl.trainer.config import _TrainingConfig

from .base import Callback


def wandb_is_available() -> bool:
    return find_spec("wandb") is not None


class WandB(Callback):  # pragma: no cover
    """
    A :class:`TrainingCallback` integrating the experiment tracking tool
    `wandb` (https://wandb.ai/).

    It allows users to store their configs, monitor their trainings
    and compare runs through a graphic interface. To be able use this feature you will need:

        - a valid `wandb` account
        - the package `wandb` installed in your virtual env. If not you can install it with

        .. code-block::

            $ pip install wandb

        - to be logged in to your wandb account using

        .. code-block::

            $ wandb login
    """

    def __init__(self):
        if not wandb_is_available():
            raise ModuleNotFoundError(
                "`wandb` package must be installed. Run `pip install wandb`"
            )

        else:
            import wandb

            self._wandb = wandb

    def setup(
        self,
        project_name: str = "clinicadl_experiment",
        entity_name: str = None,
        **kwargs,
    ):
        """
        Setup the WandbCallback.

        args:
            project_name (str): The name of the wandb project to use.

            entity_name (str): The name of the wandb entity to use.
        """

        self.is_initialized = True

        self.run = self._wandb.init(project=project_name, entity=entity_name)

        self._wandb.config.update({})

        self._wandb.define_metric("train/global_step")
        self._wandb.define_metric("*", step_metric="train/global_step", step_sync=True)

    def on_train_begin(self, config: _TrainingConfig, **kwargs):
        model_config = kwargs.pop("model_config", None)
        if not self.is_initialized:
            self.setup(training_config, model_config=model_config)

    def on_log(self, logs, **kwargs):
        global_step = kwargs.pop("global_step", None)
        logs = rename_logs(logs)

        self._wandb.log({**logs, "train/global_step": global_step})

    def on_prediction_step(self, **kwargs):
        kwargs.pop("global_step", None)

        column_names = ["images_id", "truth", "reconstruction", "normal_generation"]

        true_data = kwargs.pop("true_data", None)
        reconstructions = kwargs.pop("reconstructions", None)
        generations = kwargs.pop("generations", None)

        data_to_log = []

        if (
            true_data is not None
            and reconstructions is not None
            and generations is not None
        ):
            for i in range(len(true_data)):
                data_to_log.append(
                    [
                        f"img_{i}",
                        self._wandb.Image(
                            np.moveaxis(true_data[i].cpu().detach().numpy(), 0, -1)
                        ),
                        self._wandb.Image(
                            np.clip(
                                np.moveaxis(
                                    reconstructions[i].cpu().detach().numpy(), 0, -1
                                ),
                                0,
                                255.0,
                            )
                        ),
                        self._wandb.Image(
                            np.clip(
                                np.moveaxis(
                                    generations[i].cpu().detach().numpy(), 0, -1
                                ),
                                0,
                                255.0,
                            )
                        ),
                    ]
                )

            val_table = self._wandb.Table(data=data_to_log, columns=column_names)

            self._wandb.log({"my_val_table": val_table})

    def on_train_end(self, config: _TrainingConfig, **kwargs):
        self.run.finish()
