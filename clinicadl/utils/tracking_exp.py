"""Training Callbacks for training monitoring integrated in `pythae` (inspired from 
https://github.com/huggingface/transformers/blob/master/src/transformers/trainer_callback.py)"""

import importlib
import logging
from copy import copy
from pathlib import Path

logger = logging.getLogger(__name__)


def wandb_is_available():
    return importlib.util.find_spec("wandb")


def mlflow_is_available():
    return importlib.util.find_spec("mlflow") is not None


class Tracker:
    """
    Base class to track the metrics during training depending on the network task.
    """

    def __init__(self):
        pass

    def log_metrics(
        self,
        tracker,
        track_exp: bool = False,
        network_task: str = "classification",
        metrics_train: list = [],
        metrics_valid: list = [],
    ):
        metrics_dict = {}
        if network_task == "classification":
            metrics_dict = {
                "loss_train": metrics_train["loss"],
                "accuracy_train": metrics_train["accuracy"],
                "sensitivity_train": metrics_train["sensitivity"],
                "accuracy_train": metrics_train["accuracy"],
                "specificity_train": metrics_train["specificity"],
                "PPV_train": metrics_train["PPV"],
                "NPV_train": metrics_train["NPV"],
                "BA_train": metrics_train["BA"],
                "loss_valid": metrics_valid["loss"],
                "accuracy_valid": metrics_valid["accuracy"],
                "sensitivity_valid": metrics_valid["sensitivity"],
                "accuracy_valid": metrics_valid["accuracy"],
                "specificity_valid": metrics_valid["specificity"],
                "PPV_valid": metrics_valid["PPV"],
                "NPV_valid": metrics_valid["NPV"],
                "BA_valid": metrics_valid["BA"],
            }
        elif network_task == "reconstruction":
            metrics_dict = {
                "loss_train": metrics_train["loss"],
                "MSE_train": metrics_train["MSE"],
                "MAE_train": metrics_train["MAE"],
                "PSNR_train": metrics_train["PSNR"],
                "SSIM_train": metrics_train["SSIM"],
                "loss_valid": metrics_valid["loss"],
                "MSE_valid": metrics_valid["MSE"],
                "MAE_valid": metrics_valid["MAE"],
                "PSNR_valid": metrics_valid["PSNR"],
                "SSIM_valid": metrics_valid["SSIM"],
            }
        elif network_task == "regression":
            metrics_dict = {
                "loss_train": metrics_train["loss"],
                "MSE_train": metrics_train["MSE"],
                "MAE_train": metrics_train["MAE"],
                "loss_valid": metrics_valid["loss"],
                "MSE_valid": metrics_valid["MSE"],
                "MAE_valid": metrics_valid["MAE"],
            }

        if track_exp == "wandb":
            tracker.log(metrics_dict)
            return metrics_dict
        elif track_exp == "mlflow":
            tracker.log_metrics(metrics_dict)
            return metrics_dict


class WandB_handler(Tracker):
    def __init__(self, split: str, config: dict, maps_name: str):
        if not wandb_is_available():
            raise ModuleNotFoundError(
                "`wandb` package must be installed. Run `pip install wandb`"
            )
        else:
            import wandb

            self._wandb = wandb

        self._wandb.init(
            project="ClinicaDL",
            entity="clinicadl",
            config=config,
            save_code=True,
            group=maps_name,
            mode="online",
            name=f"split-{split}",
            reinit=True,
        )


class Mlflow_handler(Tracker):
    def __init__(self, split: str, config: dict, maps_name: str):
        if not mlflow_is_available():
            raise ModuleNotFoundError(
                "`mlflow` package must be installed. Run `pip install mlflow`"
            )
        else:
            import mlflow

            self._mlflow = mlflow

        try:
            experiment_id = self._mlflow.create_experiment(
                f"clinicadl-{maps_name}",
                artifact_location=Path.cwd().joinpath("mlruns").as_uri(),
            )

        except mlflow.exceptions.MlflowException:
            self._mlflow.set_experiment(maps_name)

        self._mlflow.start_run(experiment_id=experiment_id, run_name=f"split-{split}")
        self._mlflow.autolog()
        config_bis = copy(config)
        for cle, valeur in config.items():
            if cle == "preprocessing_dict":
                del config_bis[cle]
        config = config_bis
        self._mlflow.log_params(config)
