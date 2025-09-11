from copy import deepcopy

import pytest
import torch

from clinicadl.callbacks.factory.checkpoint_saver import _CheckpointSaver
from clinicadl.callbacks.training_state import _TrainingState
from clinicadl.model import ClinicaDLModel

from ...resources.objects import COMP, MAPS, METRICS_HANDLER, MODEL, OPTIM, SPLIT


def test_checkpoint_multiple_epochs():
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
        cs_callback.on_epoch_end(_ts)

        tmp_dir = _ts.maps.training.splits[_ts.split.index].tmp
        checkpoint = torch.load(tmp_dir.model)
        assert checkpoint["epoch"] == epoch

    cs_callback.on_train_end(_ts)
    assert not _ts.maps.training.splits[_ts.split.index].tmp.exists()


def test_model_checkpoint_content():
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
    _ts.epoch = 5
    cs_callback.on_epoch_end(_ts)

    assert _ts.split

    model_ckpt = torch.load(_ts.maps.training.splits[_ts.split.index].tmp.model)
    # assert MODEL in model_ckpt # to add
    assert "epoch" in model_ckpt
    assert model_ckpt["epoch"] == 5


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
