from clinicadl.callbacks.training_state import _TrainingState

from ..resources.objects import COMP, MAPS, METRICS, MODEL, OPTIM, SPLIT


def test_training_state():
    _ts = _TrainingState(
        maps=MAPS, metrics=METRICS, model=MODEL, optim=OPTIM, comp=COMP
    )
    assert _ts.maps == MAPS
    assert _ts.metrics == METRICS
    assert _ts.model == MODEL
    assert _ts.optim == OPTIM
    assert _ts.comp == COMP

    assert _ts.split is None
    assert _ts.epoch == -1
    assert _ts.batch == -1
    assert _ts.stop is False
    assert _ts.n_batch == -1

    _ts.reset(SPLIT)

    assert _ts.split == SPLIT
    assert _ts.epoch == 0
    assert _ts.batch == 0
    assert _ts.stop is False
    assert _ts.n_batch == 3
