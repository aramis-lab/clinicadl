# TODO: to put in trainer ? trainer_utils.py ?
from typing import Type, Union

from clinicadl.trainer.config.train import TrainConfig
from clinicadl.utils.enum import Task


def create_training_config(task: Union[str, Task]) -> Type[TrainConfig]:
    """
    A factory function to create a Training Config class suited for the task.
    Parameters
    ----------
    task : Union[str, Task]
        The Deep Learning task (e.g. classification).
    -------
    """
    task = Task(task)
    if task == Task.CLASSIFICATION:
        from clinicadl.trainer.config.classification import (
            ClassificationConfig as Config,
        )
    elif task == Task.REGRESSION:
        from clinicadl.trainer.config.regression import (
            RegressionConfig as Config,
        )
    elif task == Task.RECONSTRUCTION:
        from clinicadl.trainer.config.reconstruction import (
            ReconstructionConfig as Config,
        )
    return Config
