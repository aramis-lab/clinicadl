from copy import deepcopy

import pandas as pd
import torch

from clinicadl.callbacks.training_state import _TrainingState
from clinicadl.io.maps.training.splits import EpochTmpDir

from ..resources.objects import COMP, MAPS, METRICS_HANDLER, MODEL, OPTIM, SPLIT


def test_training_state():
    _ts = _TrainingState(
        maps=MAPS, metrics=METRICS_HANDLER, model=MODEL, optim=OPTIM, comp=COMP
    )
    assert _ts.maps == MAPS
    assert _ts.metrics == METRICS_HANDLER
    assert _ts.model == MODEL
    assert _ts.optim == OPTIM
    assert _ts.comp == COMP

    assert _ts.split is None
    assert _ts.epoch == 0
    assert _ts.batch == 0
    assert _ts.stop is False
    assert _ts.n_batch == 0

    _ts.reset(SPLIT)

    assert _ts.split == SPLIT
    assert _ts.epoch == 0
    assert _ts.batch == 0
    assert _ts.stop is False
    assert _ts.n_batch == 6  # why 6 ? maps size to check


def test_save_load_checkpoint(tmp_path):
    network = torch.nn.Linear(10, 10)
    MODEL.network = network

    chkpt_path = EpochTmpDir(tmp_path, epoch=1)
    chkpt_path.path.mkdir()

    _ts = _TrainingState(
        maps=MAPS, metrics=METRICS_HANDLER, model=MODEL, optim=OPTIM, comp=COMP
    )

    df = pd.DataFrame({"epoch": [0], "loss": [0], "mae": [0], "mse": [0]})
    detailed_df = pd.DataFrame(
        {
            "epoch": [0],
            "loss": [0],
            "mae": [0],
            "mse": [0],
            "participant_id": ["sub-001"],
            "session_id": ["ses-M000"],
        }
    )
    _ts.metrics._df = df
    _ts.metrics._detailed_df = detailed_df
    _ts.epoch = 1
    _ts.stop = True

    _ts.save_checkpoint(chkpt_path)

    MODEL.network = torch.nn.Linear(10, 10)
    _ts = _TrainingState(
        maps=MAPS,
        metrics=METRICS_HANDLER,
        model=MODEL,
        optim=OPTIM,
        comp=COMP,
        split=SPLIT,
    )
    _ts.load_checkpoint(chkpt_path)
    assert _ts.epoch == 2
    assert _ts.stop
    pd.testing.assert_frame_equal(_ts.metrics.df, df)
    pd.testing.assert_frame_equal(_ts.metrics.detailed_df, detailed_df)
    torch.testing.assert_close(MODEL.network.state_dict(), network.state_dict())
