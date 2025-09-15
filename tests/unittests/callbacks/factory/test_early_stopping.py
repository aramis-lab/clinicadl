import pandas as pd
import pytest

from clinicadl.callbacks.factory.early_stopping import (
    EarlyStopping,
    OneMetricEarlyStopping,
)
from clinicadl.callbacks.training_state import _TrainingState

from ...resources.objects import COMP, MAPS, METRICS_HANDLER, MODEL, OPTIM, SPLIT

GOOD_INPUTS = [
    (["mae", "loss"], 5, 0.0, "min", True, None, None),
    (
        ["mae", "loss"],
        [3, 7],
        [0.0, 0.1],
        ["min", "max"],
        [True, False],
        [None, None],
        [None, None],
    ),
    (["mae", "mse"], [3, 7], 0.1, ["min", "max"], True, [None, None], [None, None]),
]

metrics_df = pd.DataFrame(
    {
        "epoch": [0, 1, 2, 3],
        "mae": [0.1, 0.2, 0.3, 0.4],
        "loss": [0.1, 0.2, 0.3, 0.4],
        "mse": [0.1, 0.2, 0.3, 0.4],
    }
)
metrics_df.set_index("epoch", inplace=True)


@pytest.mark.parametrize(
    "metrics,patience,min_delta,mode,check_finite,upper_bound,lower_bound", GOOD_INPUTS
)
def test_good_inputs(
    metrics, patience, min_delta, mode, check_finite, upper_bound, lower_bound
):
    es_ = EarlyStopping(
        metrics=metrics,
        patience=patience,
        min_delta=min_delta,
        mode=mode,
        check_finite=check_finite,
        upper_bound=upper_bound,
        lower_bound=lower_bound,
    )
    assert isinstance(es_.patience, list)
    assert isinstance(es_.min_delta, list)
    assert isinstance(es_.mode, list)
    assert isinstance(es_.check_finite, list)
    assert isinstance(es_.upper_bound, list)
    assert isinstance(es_.lower_bound, list)

    for oes_ in es_.early_config_list:
        assert isinstance(oes_.patience, int)
        assert isinstance(oes_.min_delta, float)
        assert isinstance(oes_.mode, str)
        assert isinstance(oes_.check_finite, bool)
        assert isinstance(oes_, OneMetricEarlyStopping)

    _ts = _TrainingState(
        maps=MAPS, metrics=METRICS_HANDLER, model=MODEL, optim=OPTIM, comp=COMP
    )
    _ts.reset(SPLIT)
    _ts.metrics._df = metrics_df

    es_.on_epoch_end(_ts)
    assert not _ts.stop

    # TODO : NEED TO ADD TEST WHEN EARLY STOPPING IS TRIGGERED


def test_bad_inputs():
    with pytest.raises(ValueError):
        EarlyStopping(
            metrics=["mae"],
            patience=3,
            min_delta=0.1,
            mode="min",
            check_finite=True,
            upper_bound=0.2,
            lower_bound=0.3,
        )

    with pytest.raises(ValueError):
        EarlyStopping(
            metrics=["mae"],
            patience=3,
            min_delta=0.1,
            mode="minmax",
        )

    es = EarlyStopping(metrics=["mae"])

    _ts = _TrainingState(
        maps=MAPS, metrics=METRICS_HANDLER, model=MODEL, optim=OPTIM, comp=COMP
    )
    _ts.reset(SPLIT)

    with pytest.raises(ValueError):
        _ts.metrics.df.loc[_ts.epoch, "mae"] = pd.NA
        es.on_epoch_end(_ts)

    with pytest.raises(ValueError):
        _ts.metrics.df.loc[_ts.epoch, "mae"] = pd.NA
        es.on_epoch_end(_ts)

    with pytest.raises(ValueError):
        es.on_epoch_end(_ts)
