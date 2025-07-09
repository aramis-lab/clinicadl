import torch

from clinicadl.callbacks.factory.checkpoint_saver import _CheckpointSaver
from clinicadl.callbacks.training_state import _TrainingState

from ...resources.objects import COMP, MAPS, METRICS, MODEL, OPTIM, SPLIT


def check_():
    cs_callback = _CheckpointSaver()

    _ts = _TrainingState(
        maps=MAPS, metrics=METRICS, model=MODEL, optim=OPTIM, comp=COMP
    )
    _ts.reset(SPLIT)

    cs_callback.on_train_begin(_ts)
    cs_callback.on_epoch_begin(_ts)

    cs_callback.on_epoch_end(_ts)
    cs_callback.on_train_end(_ts)
