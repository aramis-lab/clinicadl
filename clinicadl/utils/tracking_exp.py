"""Training Callbacks for training monitoring integrated in `pythae` (inspired from 
https://github.com/huggingface/transformers/blob/master/src/transformers/trainer_callback.py)"""

import importlib
import logging
import os
import sys

logger = logging.getLogger(__name__)


def wandb_is_available():
    return importlib.util.find_spec("wandb")


def mlflow_is_available():
    return importlib.util.find_spec("mlflow") is not None


class WandB_class:
    def __init__(self):
        if not wandb_is_available():
            raise ModuleNotFoundError(
                "`wandb` package must be installed. Run `pip install wandb`"
            )
        else:
            import wandb

            self._wandb = wandb


#     def setup(
#         self,
#         training_config: BaseTrainerConfig,
#         model_config: BaseAEConfig = None,
#         project_name: str = "ClinicaDL",
#         entity_name: str = None,
#         **kwargs,
#     ):
#         """
#         Setup the WandbCallback.

#         args:
#             training_config (BaseTrainerConfig): The training configuration used in the run.

#             model_config (BaseAEConfig): The model configuration used in the run.

#             project_name (str): The name of the wandb project to use.

#             entity_name (str): The name of the wandb entity to use.
#         """

#         self.is_initialized = True

#         training_config_dict = training_config.to_dict()

#         self.run = self._wandb.init(project=project_name, entity=entity_name)

#         if model_config is not None:
#             model_config_dict = model_config.to_dict()

#             self._wandb.config.update(
#                 {
#                     "training_config": training_config_dict,
#                     "model_config": model_config_dict,
#                 }
#             )

#         else:
#             self._wandb.config.update({**training_config_dict})

#         self._wandb.define_metric("train/global_step")
#         self._wandb.define_metric("*", step_metric="train/global_step", step_sync=True)

#     def on_train_begin(self, training_config: BaseTrainerConfig, **kwargs):
#         model_config = kwargs.pop("model_config", None)
#         if not self.is_initialized:
#             self.setup(training_config, model_config=model_config)

#     def on_log(self, training_config: BaseTrainerConfig, logs, **kwargs):
#         global_step = kwargs.pop("global_step", None)
#         logs = rename_logs(logs)

#         self._wandb.log({**logs, "train/global_step": global_step})

#     def on_prediction_step(self, training_config: BaseTrainerConfig, **kwargs):
#         kwargs.pop("global_step", None)

#         column_names = ["images_id", "truth", "reconstruction", "normal_generation"]

#         true_data = kwargs.pop("true_data", None)
#         reconstructions = kwargs.pop("reconstructions", None)
#         generations = kwargs.pop("generations", None)

#         data_to_log = []

#         if (
#             true_data is not None
#             and reconstructions is not None
#             and generations is not None
#         ):
#             for i in range(len(true_data)):

#                 data_to_log.append(
#                     [
#                         f"img_{i}",
#                         self._wandb.Image(
#                             np.moveaxis(true_data[i].cpu().detach().numpy(), 0, -1)
#                         ),
#                         self._wandb.Image(
#                             np.clip(
#                                 np.moveaxis(
#                                     reconstructions[i].cpu().detach().numpy(), 0, -1
#                                 ),
#                                 0,
#                                 255.0,
#                             )
#                         ),
#                         self._wandb.Image(
#                             np.clip(
#                                 np.moveaxis(
#                                     generations[i].cpu().detach().numpy(), 0, -1
#                                 ),
#                                 0,
#                                 255.0,
#                             )
#                         ),
#                     ]
#                 )

#             val_table = self._wandb.Table(data=data_to_log, columns=column_names)

#             self._wandb.log({"my_val_table": val_table})

#     def on_train_end(self, training_config: BaseTrainerConfig, **kwargs):
#         self.run.finish()


# class MLFlowCallback(TrainingCallback):  # pragma: no cover
#     """
#     A :class:`TrainingCallback` integrating the experiment tracking tool
#     `mlflow` (https://mlflow.org/).

#     It allows users to store their configs, monitor their trainings
#     and compare runs through a graphic interface. To be able use this feature you will need:

#         - the package `mlfow` installed in your virtual env. If not you can install it with

#         .. code-block::

#             $ pip install mlflow
#     """

#     def __init__(self):
#         if not mlflow_is_available():
#             raise ModuleNotFoundError(
#                 "`mlflow` package must be installed. Run `pip install mlflow`"
#             )

#         else:
#             import mlflow

#             self._mlflow = mlflow

#     def setup(
#         self,
#         training_config: BaseTrainerConfig,
#         model_config: BaseAEConfig = None,
#         run_name: str = None,
#         **kwargs,
#     ):
#         """
#         Setup the MLflowCallback.

#         args:
#             training_config (BaseTrainerConfig): The training configuration used in the run.

#             model_config (BaseAEConfig): The model configuration used in the run.

#             run_name (str): The name to apply to the current run.
#         """
#         self.is_initialized = True

#         training_config_dict = training_config.to_dict()

#         self._mlflow.start_run(run_name=run_name)

#         logger.info(
#             f"MLflow run started with run_id={self._mlflow.active_run().info.run_id}"
#         )
#         if model_config is not None:
#             model_config_dict = model_config.to_dict()

#             self._mlflow.log_params(
#                 {
#                     **training_config_dict,
#                     **model_config_dict,
#                 }
#             )

#         else:
#             self._mlflow.log_params({**training_config_dict})

#     def on_train_begin(self, training_config, **kwargs):
#         model_config = kwargs.pop("model_config", None)
#         if not self.is_initialized:
#             self.setup(training_config, model_config=model_config)

#     def on_log(self, training_config: BaseTrainerConfig, logs, **kwargs):
#         global_step = kwargs.pop("global_step", None)

#         logs = rename_logs(logs)
#         metrics = {}
#         for k, v in logs.items():
#             if isinstance(v, (int, float)):
#                 metrics[k] = v

#         self._mlflow.log_metrics(metrics=metrics, step=global_step)

#     def on_train_end(self, training_config: BaseTrainerConfig, **kwargs):
#         self._mlflow.end_run()

#     def __del__(self):
#         # if the previous run is not terminated correctly, the fluent API will
#         # not let you start a new run before the previous one is killed
#         if (
#             callable(getattr(self._mlflow, "active_run", None))
#             and self._mlflow.active_run() is not None
#         ):
#             self._mlflow.end_run()
