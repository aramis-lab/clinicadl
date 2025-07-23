import pytest

from clinicadl.callbacks.factory.checkpoint import Checkpoint
from clinicadl.callbacks.factory.checkpoint_saver import _CheckpointSaver
from clinicadl.callbacks.training_state import _TrainingState

from ...resources.objects import COMP, MAPS, METRICS_HANDLER, MODEL, OPTIM, SPLIT

GOOD_INPUTS = [
    (5, None),
    (6, [3, 7]),
    (1, [2, 4, 6]),
    (3, [1, 2]),
    (11, None),
]


@pytest.mark.parametrize("patience, epochs", GOOD_INPUTS)
def test_good_checkpoint(patience, epochs):
    checkpoint = Checkpoint(patience=patience, epochs=epochs)
    _saver = _CheckpointSaver()

    OPTIM.epochs = 8
    MAPS.load()
    _ts = _TrainingState(
        maps=MAPS, metrics=METRICS_HANDLER, model=MODEL, optim=OPTIM, comp=COMP
    )
    _ts.reset(SPLIT)

    for epoch in range(OPTIM.epochs):
        _ts.epoch = epoch
        _saver.on_train_begin(_ts)
        _saver.on_epoch_begin(_ts)

        _saver.on_epoch_end(_ts)
        checkpoint.on_epoch_end(_ts)

        if (
            epoch in (epochs if epochs else [])
            or epoch % patience == 0
            or epoch == OPTIM.epochs - 1
        ):
            assert _ts.split is not None
            assert (
                _ts.maps.training.splits[_ts.split.index]
                .checkpoints.epochs[epoch]
                .exists()
            )
            assert _ts.maps.training.splits[_ts.split.index].tmp.model.is_file()
            assert _ts.maps.training.splits[_ts.split.index].tmp.optimizer.is_file()

    _saver.on_train_end(_ts)
    checkpoint.on_train_end(_ts)

    assert _ts.split is not None
    assert not _ts.maps.training.splits[_ts.split.index].tmp.exists()


def test_bad_checkpoint():
    with pytest.raises(ValueError):
        Checkpoint(patience=0)

    with pytest.raises(ValueError):
        Checkpoint(patience=-1)
