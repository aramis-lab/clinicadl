import os
from copy import deepcopy

import pytest

from clinicadl.callbacks.factory.checkpoint_saver import _CheckpointSaver
from clinicadl.callbacks.factory.lr_scheduler import LRScheduler
from clinicadl.callbacks.handler import _CallbacksHandler
from clinicadl.callbacks.training_state import _TrainingState
from clinicadl.models.base import ClinicaDLModel

from ...resources.objects import (
    COMP,
    LR_SCHEDULER,
    MAPS,
    METRICS_HANDLER,
    MODEL,
    NETWORK,
    OPTIM,
    OPTIMIZER,
    SPLIT,
)

MAPS.load()


def test_checkpoint_multiple_epochs():
    optim = OPTIMIZER.get_object(network=NETWORK.get_object())
    callbacks = _CallbacksHandler(
        metrics=METRICS_HANDLER,
        callbacks=[
            LRScheduler(scheduler=LR_SCHEDULER, optimizer=optim),
            LRScheduler(scheduler=LR_SCHEDULER, optimizer=optim),
        ],
    )
    cs_callback = _CheckpointSaver()
    _ts = _TrainingState(
        maps=deepcopy(MAPS),
        metrics=METRICS_HANDLER,
        model=MODEL,
        optim=OPTIM,
        comp=COMP,
    )
    _ts.reset(deepcopy(SPLIT))
    cs_callback.on_train_begin(_ts)

    assert _ts.split

    for epoch in range(3):
        _ts.epoch = epoch
        cs_callback.on_epoch_end(_ts, callbacks=callbacks)

        tmp_dir = _ts.maps.training.splits[_ts.split.index].tmp(epoch)
        assert os.listdir(tmp_dir.path) == ["callbacks", "model.pth.tar"]
        assert os.listdir(tmp_dir.path / "callbacks") == [
            "_training_loss.tsv",
            "lr_scheduler.pt",
            "lr_scheduler_2.pt",
        ]

        assert (
            not _ts.maps.training.splits[_ts.split.index].tmp(epoch=epoch - 1).exists()
        )

    cs_callback.on_train_end(_ts)
    assert not _ts.maps.training.splits[_ts.split.index].tmp.exists()


def test_bad_checkpoint():
    cs_callback = _CheckpointSaver()

    _ts = _TrainingState(
        maps=deepcopy(MAPS),
        metrics=METRICS_HANDLER,
        model=MODEL,
        optim=OPTIM,
        comp=COMP,
    )

    with pytest.raises(ValueError):
        cs_callback.on_train_begin(_ts)

    _ts.reset(deepcopy(SPLIT))
    assert _ts.split
    _ts.split.train_loader = None

    with pytest.raises(ValueError):
        cs_callback.on_train_begin(_ts)
