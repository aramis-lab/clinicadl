import pandas as pd
import pytest

from clinicadl.callbacks.factory.model_selection import ModelSelection
from clinicadl.callbacks.training_state import _TrainingState

from ...resources.objects import COMP, MAPS, METRICS_HANDLER, MODEL, OPTIM, SPLIT

metrics_df = pd.DataFrame(
    {
        "epoch": [0, 1, 2, 3],
        "mae": [0.1, 0.2, 0.3, 0.4],
        "loss": [0.1, 0.2, 0.3, 0.4],
        "mse": [0.1, 0.2, 0.3, 0.4],
    }
)
metrics_df.set_index("epoch", inplace=True)


def test_good_inputs():
    ms_callback = ModelSelection(metrics=["mae"])
    _ts = _TrainingState(
        maps=MAPS, metrics=METRICS_HANDLER, model=MODEL, optim=OPTIM, comp=COMP
    )
    _ts.reset(SPLIT)

    for metric in _ts.metrics.metrics:
        assert _ts.split is not None
        assert _ts.maps.training.splits[_ts.split.index].best_metrics_list == []

        ms_callback.on_train_begin(_ts)
        assert _ts.maps.training.splits[_ts.split.index].best_metrics[metric].exists()

        ms_callback.on_train_end(_ts)
