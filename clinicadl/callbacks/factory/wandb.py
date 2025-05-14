from importlib.util import find_spec

from .base import Callback


def wandb_is_available() -> bool:
    return find_spec("wandb") is not None


class WandBCallback(Callback):
    def __init__(self):
        if not wandb_is_available():
            raise ModuleNotFoundError(
                "`wandb` package must be installed. Run `pip install wandb`"
            )
        else:
            import wandb

            self._wandb = wandb

    def on_train_begin(self, **kwargs):
        # OLD CODE FOR WANDB
        # self._wandb.init(
        #     project="ClinicaDL",
        #     entity="clinicadl",
        #     config=config,
        #     save_code=True,
        #     group=maps_name,
        #     mode="online",
        #     name=f"split-{split}",
        #     reinit=True,
        # )
        pass

    def on_train_end(self, **kwargs):
        pass
