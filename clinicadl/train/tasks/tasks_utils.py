from typing import Type, Union

from clinicadl.train.trainer import Task, TrainingConfig


def create_training_config(task: Union[str, Task]) -> Type[TrainingConfig]:
    """
    A factory function to create a Training Config class suited for the task.

    Parameters
    ----------
    task : Union[str, Task]
        The Deep Learning task (e.g. classification).

    Returns
    -------
    Type[TrainingConfig]
        The Config class.
    """
    task = Task(task)
    if task == Task.CLASSIFICATION:
        from .classification import ClassificationConfig as Config
    elif task == Task.REGRESSION:
        from .regression import RegressionConfig as Config
    elif task == Task.RECONSTRUCTION:
        from .reconstruction import ReconstructionConfig as Config
    return Config
