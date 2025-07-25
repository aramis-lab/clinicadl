# TODO : Not working at the moment


from importlib.util import find_spec
from typing import Optional

import numpy as np

from .base import Callback


class Comet(Callback):
    """
    A :class:`TrainingCallback` integrating the experiment tracking tool `comet_ml <https://www.comet.com/site/>`_.

    It allows users to store their configs, monitor
    their trainings and compare runs through a graphic interface. To be able use this feature
    you will need:

    - the package `comet_ml` installed in your virtual env. If not you can install it with

    .. code-block::

        $ pip install comet_ml
    """

    def __init__(self):
        if not self.is_available():
            raise ModuleNotFoundError(
                "`comet_ml` package must be installed. Run `pip install comet_ml`"
            )

        else:
            import comet_ml

            self._comet_ml = comet_ml
            self.is_initialized = False

    @staticmethod
    def is_available():
        """TO COMPLETE"""
        return find_spec("comet_ml") is not None

    def setup(
        self,
        api_key: Optional[str] = None,
        project_name: str = "clinicadl_experiment",
        workspace: Optional[str] = None,
        offline_run: bool = False,
        offline_directory: str = "./",
        **kwargs,
    ):
        """
        Setup the CometCallback.

        args:
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
            experiment.log_other("Created from", "clinicadl")
        else:
            experiment = self._comet_ml.OfflineExperiment(
                api_key=api_key,
                project_name=project_name,
                workspace=workspace,
                offline_directory=offline_directory,
            )
            experiment.log_other("Created from", "clinicadl")

        # experiment.log_parameters({}, prefix="training_config/")
        # experiment.log_parameters({}, prefix="model_config/")
