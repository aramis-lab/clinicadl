from typing import Type, Union

from clinicadl.config.config.pipelines.train import TrainConfig
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
        from clinicadl.config.config.pipelines.task.classification import (
            ClassificationConfig as Config,
        )
    elif task == Task.REGRESSION:
        from clinicadl.config.config.pipelines.task.regression import (
            RegressionConfig as Config,
        )
    elif task == Task.RECONSTRUCTION:
        from clinicadl.config.config.pipelines.task.reconstruction import (
            ReconstructionConfig as Config,
        )
    return Config
