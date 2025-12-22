import os
import shutil
from copy import deepcopy

import pytest

from clinicadl.callbacks.handler import _CallbacksHandler
from clinicadl.callbacks.implemented.checkpoint_saver import _CheckpointSaver
from clinicadl.callbacks.implemented.lr_scheduler import LRScheduler
from clinicadl.io.maps import Maps
from clinicadl.train.trainer_state import TrainerState

from ...resources.objects import (
    COMP,
    LR_SCHEDULER,
    MAPS_DIR,
    METRICS_HANDLER,
    MODEL,
    NETWORK,
    OPTIM,
    OPTIMIZER,
    SPLIT,
)


def test_checkpoint_multiple_epochs(tmp_path):
    shutil.copytree(MAPS_DIR, tmp_path / "maps")
    MAPS = Maps(tmp_path / "maps")
    MAPS.read()
    MAPS.training.splits[1].best_models["loss"].remove()

    optim = OPTIMIZER.get_object(network=NETWORK.get_object())
    callbacks = _CallbacksHandler(
        metrics=METRICS_HANDLER,
        callbacks=[
            LRScheduler(scheduler=LR_SCHEDULER, optimizer=optim),
            LRScheduler(scheduler=LR_SCHEDULER, optimizer=optim),
        ],
    )
    cs_callback = _CheckpointSaver()
    _ts = TrainerState(
        maps=deepcopy(MAPS),
        metrics=METRICS_HANDLER,
        model=MODEL,
        optim=OPTIM,
        comp=COMP,
    )
    _ts.reset(deepcopy(SPLIT))
    callbacks.on_train_begin(_ts)
    cs_callback.on_train_begin(_ts)

    assert _ts.split

    for epoch in range(3):
        _ts.epoch = epoch
        cs_callback.on_epoch_end(_ts, callbacks=callbacks)

        tmp_dir = _ts.maps.training.splits[_ts.split.index].tmp.epochs[epoch]
        assert set(os.listdir(tmp_dir.path)) == set(
            ["metrics", "callbacks", "model.pth.tar", "stop.json"]
        )
        assert set(os.listdir(tmp_dir.path / "callbacks")) == set(
            [
                "_training_loss.tsv",
                "lr_scheduler.pt",
                "lr_scheduler_2.pt",
            ]
        )

        if epoch > 0:
            assert (
                not _ts.maps.training.splits[_ts.split.index]
                .tmp.epochs[epoch - 1]
                .exists()
            )

    cs_callback.on_train_end(_ts)
    assert not _ts.maps.training.splits[_ts.split.index].tmp.exists()


def test_bad_checkpoint(tmp_path):
    shutil.copytree(MAPS_DIR, tmp_path / "maps")
    MAPS = Maps(tmp_path / "maps")
    MAPS.read()

    cs_callback = _CheckpointSaver()

    _ts = TrainerState(
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
