from importlib.util import find_spec
from typing import Union

import pandas as pd

from clinicadl.callbacks.training_state import _TrainingState
from clinicadl.dictionary.suffixes import PTH, TAR
from clinicadl.dictionary.words import MODEL, OPTIMIZER

from .base import Callback


class WandB(Callback):  # pragma: no cover
    """
    A training callback that integrates with the experiment tracking tool
    `Weights & Biases <https://wandb.ai/>`_.

    This callback enables logging of training configurations, metrics, and artifacts,
    allowing users to monitor experiments and compare training runs through the WandB
    web interface.

    Requirements
    ------------
        - The `wandb` package must be installed in your Python environment.
        You can install it with:

    .. code-block:: bash

        pip install wandb

    .. note::
        - WandB supports local and cloud logging.
        - This callback is useful for reproducibility and experiment tracking.

    Examples
    --------
    .. code-block:: python

        from clinicadl.callbacks import WandB

        wandb_callback = WandB(project="my_project", entity="my_team")
        handler = _CallbacksHandler(callbacks=[wandb_callback])
    """

    def __init__(
        self,
        project: str = "default_project",
        entity: Union[str, None] = None,
        run_name: Union[str, None] = None,
        config: dict = None,
    ):
        if not self.is_available():
            raise ModuleNotFoundError(
                "`wandb` package must be installed. Run `pip install wandb`"
            )

        else:
            import wandb

            self._wandb = wandb

        self.project = project
        self.entity = entity
        self.run_name = run_name
        self.config = config or {}
        self.run = None

    @staticmethod
    def is_available() -> bool:
        return find_spec("wandb") is not None

    def on_train_begin(self, config: _TrainingState, **kwargs) -> None:
        # Optionally update config with hyperparameters from training config
        training_config = {}
        # Example: add learning rate or epochs if present
        if hasattr(config, MODEL) and hasattr(config.model, OPTIMIZER):
            for param_group in config.model.optimizer.param_groups:
                for k, v in param_group.items():
                    if isinstance(v, (int, float, str)):
                        training_config[k] = v
        training_config.update(self.config)

        self.run = self._wandb.init(
            project=self.project,
            entity=self.entity,
            name=self.run_name,
            config=training_config,
            reinit=True,
        )

    def on_epoch_end(self, config: _TrainingState, **kwargs) -> None:
        if config.metrics.df is not None and not config.metrics.df.empty:
            if config.epoch in config.metrics.df.index:
                epoch_metrics = config.metrics.df.loc[config.epoch]
                if epoch_metrics is not None:
                    log_dict = {
                        metric_name: float(value)
                        for metric_name, value in epoch_metrics.items()
                        if value is not None and not pd.isna(value)
                    }
                    self._wandb.log(log_dict, step=config.epoch)

    def on_train_end(self, config: _TrainingState, **kwargs) -> None:
        tmp_dir = config.maps.training.splits[config.split.index].tmp
        model_file = tmp_dir.model
        optimizer_file = tmp_dir.optimizer

        if model_file.exists():
            self._wandb.save(str(model_file), base_path=str(tmp_dir.path), policy="now")

        if optimizer_file.exists():
            self._wandb.save(
                str(optimizer_file), base_path=str(tmp_dir.path), policy="now"
            )

        self._wandb.finish()
