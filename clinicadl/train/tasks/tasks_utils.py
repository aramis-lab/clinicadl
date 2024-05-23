from typing import Type, Union

from clinicadl.train.trainer import Task, TrainingConfig


def create_training_config(task: Union[str, Task]) -> Type[TrainingConfig]:
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
        from clinicadl.train.tasks.classification.config import (
            ClassificationConfig as Config,
        )
    elif task == Task.REGRESSION:
        from clinicadl.train.tasks.regression.config import RegressionConfig as Config
    elif task == Task.RECONSTRUCTION:
        from clinicadl.train.tasks.reconstruction.config import (
            ReconstructionConfig as Config,
        )
    return Config
