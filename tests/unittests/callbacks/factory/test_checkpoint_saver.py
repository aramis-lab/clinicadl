import pytest

from clinicadl.callbacks.factory.checkpoint_saver import _CheckpointSaver
from clinicadl.callbacks.training_state import _TrainingState

from ...resources.objects import COMP, MAPS, METRICS_HANDLER, MODEL, OPTIM, SPLIT


def test_good_checkpoint():
    cs_callback = _CheckpointSaver()

    _ts = _TrainingState(
        maps=MAPS, metrics=METRICS_HANDLER, model=MODEL, optim=OPTIM, comp=COMP
    )
    _ts.reset(SPLIT)

    cs_callback.on_train_begin(_ts)
    cs_callback.on_epoch_begin(_ts)

    cs_callback.on_epoch_end(_ts)
    assert _ts.split is not None
    assert _ts.maps.training.splits[_ts.split.index].tmp.exists()
    assert _ts.maps.training.splits[_ts.split.index].tmp.model.is_file()
    assert _ts.maps.training.splits[_ts.split.index].tmp.optimizer.is_file()

    cs_callback.on_train_end(_ts)
    assert not _ts.maps.training.splits[_ts.split.index].tmp.exists()


def test_bad_checkpoint():
    cs_callback = _CheckpointSaver()

    _ts = _TrainingState(
        maps=MAPS, metrics=METRICS_HANDLER, model=MODEL, optim=OPTIM, comp=COMP
    )

    with pytest.raises(ValueError):
        cs_callback.on_train_begin(_ts)
