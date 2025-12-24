import logging
import re

import numpy as np
import pandas as pd
import pytest

from clinicadl.callbacks import EarlyStoppingCallback
from clinicadl.metrics.config import (
    MAEMetricConfig,
    MSEMetricConfig,
    PSNRMetricConfig,
)
from clinicadl.train.trainer_state import TrainerState

MSE = MSEMetricConfig().get_object()
MAE = MAEMetricConfig().get_object()
PSNR = PSNRMetricConfig(max_val=1).get_object()

METRICS = pd.DataFrame(
    {
        "epoch": [0, 2, 4, 6, 9],
        "mae": [np.inf, np.nan, 10.1, -10.1, -np.inf],
        "mse": [-1.0, -1.11, -1.2, -5, -10],
        "mse_overfit": [-1.0, -1.11, -1.2, -1.11, -1.0],
        "psnr": [1.0, 0.8, 1.11, 1.20, 0.16],
        "bad": ["a", "b", "c", "d", "e"],
    }
)
LOGGER = logging.getLogger(__name__)


def test_inputs():
    es = EarlyStoppingCallback(
        metric=["psnr", "mse"],
        patience=5,
        min_delta=(0.0, 0.1),
        check_finite=True,
        upper_bound=None,
        lower_bound=0.1,
    )
    es.on_train_start()
    es.on_validation_start(metrics={"psnr": PSNR, "mse": MSE})
    assert (es.stoppers[0].config.metric, es.stoppers[1].config.metric) == (
        "psnr",
        "mse",
    )
    assert (es.stoppers[0].config.patience, es.stoppers[1].config.patience) == (5, 5)
    assert (es.stoppers[0].config.min_delta, es.stoppers[1].config.min_delta) == (
        0.0,
        0.1,
    )
    assert (es.stoppers[0].config.mode, es.stoppers[1].config.mode) == ("max", "min")
    assert (es.stoppers[0].config.check_finite, es.stoppers[1].config.check_finite) == (
        True,
        True,
    )
    assert (es.stoppers[0].config.upper_bound, es.stoppers[1].config.upper_bound) == (
        None,
        None,
    )
    assert (es.stoppers[0].config.lower_bound, es.stoppers[1].config.lower_bound) == (
        0.1,
        0.1,
    )

    es = EarlyStoppingCallback(
        metric=["psnr", "mse"],
        patience=[3, 7],
        min_delta=0,
        check_finite=[True, False],
        upper_bound=[None, 0.1],
        lower_bound=[None, None],
    )
    es.on_train_start()
    with pytest.raises(KeyError, match="'psnr' not found in the validation metrics!"):
        es.on_validation_start(metrics={"mse": MSE})
    es.on_validation_start(metrics={"psnr": PSNR, "mse": MSE})

    assert (es.stoppers[0].config.patience, es.stoppers[1].config.patience) == (3, 7)
    assert (es.stoppers[0].config.min_delta, es.stoppers[1].config.min_delta) == (
        0.0,
        0.0,
    )
    assert (es.stoppers[0].config.check_finite, es.stoppers[1].config.check_finite) == (
        True,
        False,
    )
    assert (es.stoppers[0].config.upper_bound, es.stoppers[1].config.upper_bound) == (
        None,
        0.1,
    )
    assert (es.stoppers[0].config.lower_bound, es.stoppers[1].config.lower_bound) == (
        None,
        None,
    )

    with pytest.raises(
        ValueError,
        match=re.escape(
            "For EarlyStoppingCallback, there are 2 metrics, but you passed 3 'patience': [1, 2, 3]"
        ),
    ):
        EarlyStoppingCallback(metric=["psnr", "mse"], patience=[1, 2, 3])


def test_numeric():
    early_stopping = EarlyStoppingCallback(metric="bad")
    early_stopping.on_train_start()
    early_stopping.on_validation_start(metrics={"bad": PSNR})
    with pytest.raises(
        ValueError, match="Value for metric 'bad' at epoch 0 is not numeric."
    ):
        early_stopping.on_validation_end(state=TrainerState(), metrics_df=METRICS)


def test_one_metric(caplog):
    state = TrainerState()
    early_stopping = EarlyStoppingCallback(
        metric="mae", check_finite=True, upper_bound=10, lower_bound=-10
    )

    state.should_stop = False
    state.current_epoch = 0
    early_stopping.on_validation_end(state=state, metrics_df=METRICS)
    assert not state.should_stop

    early_stopping.on_train_start()
    early_stopping.on_validation_start(metrics={"mae": MAE})

    state.should_stop = False
    state.current_epoch = 0
    with caplog.at_level(logging.WARNING):
        early_stopping.on_validation_end(state=state, metrics_df=METRICS)
    assert (
        "Metric 'mae' value at epoch 0 is not a finite float. Stopping training."
        in caplog.text
    )
    assert state.should_stop

    state.should_stop = False
    state.current_epoch = 2
    with caplog.at_level(logging.WARNING):
        early_stopping.on_validation_end(state=state, metrics_df=METRICS)
    assert (
        "Metric 'mae' value at epoch 2 is not a finite float. Stopping training."
        in caplog.text
    )
    assert state.should_stop

    state.should_stop = False
    state.current_epoch = 4
    with caplog.at_level(logging.WARNING):
        early_stopping.on_validation_end(state=state, metrics_df=METRICS)
    assert (
        "Metric 'mae' value 10.1 exceeds upper bound 10.0 at epoch 4. Stopping training."
        in caplog.text
    )
    assert state.should_stop

    state.should_stop = False
    state.current_epoch = 6
    with caplog.at_level(logging.WARNING):
        early_stopping.on_validation_end(state=state, metrics_df=METRICS)
    assert (
        "Metric 'mae' value -10.1 falls below lower bound -10.0 at epoch 6. Stopping training."
        in caplog.text
    )
    assert state.should_stop

    state.should_stop = False
    state.current_epoch = 9
    with caplog.at_level(logging.WARNING):
        early_stopping.on_validation_end(state=state, metrics_df=METRICS)
    assert (
        "Metric 'mae' value at epoch 2 is not a finite float. Stopping training."
        in caplog.text
    )
    assert state.should_stop

    state = TrainerState()
    early_stopping = EarlyStoppingCallback(metric="psnr", patience=2, min_delta=0.1)
    early_stopping.on_train_start()
    early_stopping.on_validation_start(metrics={"psnr": PSNR})

    state.should_stop = False
    state.current_epoch = 0
    early_stopping.on_validation_end(state=state, metrics_df=METRICS)
    assert early_stopping.stoppers[0].best == 1.0
    assert not state.should_stop

    state.current_epoch = 2
    with caplog.at_level(logging.DEBUG):
        early_stopping.on_validation_end(state=state, metrics_df=METRICS)
    assert "No improvement in 'psnr' for 1 evaluation step(s)." in caplog.text
    assert not state.should_stop

    state.current_epoch = 4
    early_stopping.on_validation_end(state=state, metrics_df=METRICS)
    assert not state.should_stop

    state.current_epoch = 6
    with caplog.at_level(logging.DEBUG):
        early_stopping.on_validation_end(state=state, metrics_df=METRICS)
    assert "No improvement in 'psnr' for 1 evaluation step(s)." in caplog.text
    assert not state.should_stop

    state.current_epoch = 9
    with caplog.at_level(logging.INFO):
        early_stopping.on_validation_end(state=state, metrics_df=METRICS)
    assert (
        "Early stopping triggered on metric 'psnr' after 2 evaluation(s) without improvement."
        in caplog.text
    )
    assert (
        "Early stopping criteria met for all monitored metrics. Stopping training."
        in caplog.text
    )
    assert state.should_stop

    state = TrainerState()
    early_stopping = EarlyStoppingCallback(metric="mse", patience=1, min_delta=0.1)
    early_stopping.on_train_start()
    early_stopping.on_validation_start(metrics={"mse": MSE})

    state.should_stop = False
    state.current_epoch = 0
    early_stopping.on_validation_end(state=state, metrics_df=METRICS)
    assert not state.should_stop

    state.current_epoch = 2
    early_stopping.on_validation_end(state=state, metrics_df=METRICS)
    assert not state.should_stop

    state.current_epoch = 4
    early_stopping.on_validation_end(state=state, metrics_df=METRICS)
    assert state.should_stop


def test_mutiple_metrics(caplog):
    state = TrainerState()
    early_stopping = EarlyStoppingCallback(
        metric=["psnr", "mse"], patience=[2, 1], min_delta=0.1
    )
    early_stopping.on_train_start()
    early_stopping.on_validation_start(metrics={"mse": MSE, "psnr": PSNR})

    state.should_stop = False
    state.current_epoch = 0
    early_stopping.on_validation_end(state=state, metrics_df=METRICS)
    assert not state.should_stop

    state.current_epoch = 2
    early_stopping.on_validation_end(state=state, metrics_df=METRICS)
    assert not state.should_stop

    state.current_epoch = 4
    with caplog.at_level(logging.INFO):
        early_stopping.on_validation_end(state=state, metrics_df=METRICS)
    assert (
        "Early stopping triggered on metric 'mse' after 1 evaluation(s) without improvement."
        in caplog.text
    )
    assert not state.should_stop

    state.current_epoch = 6
    early_stopping.on_validation_end(state=state, metrics_df=METRICS)
    assert not state.should_stop

    state.current_epoch = 9
    with caplog.at_level(logging.INFO):
        early_stopping.on_validation_end(state=state, metrics_df=METRICS)
    assert (
        "Early stopping triggered on metric 'psnr' after 2 evaluation(s) without improvement."
        in caplog.text
    )
    assert not state.should_stop  # loss restarted to decrease

    ###
    state = TrainerState()
    early_stopping = EarlyStoppingCallback(
        metric=["psnr", "mse_overfit"],
        patience=[2, 1],
        min_delta=0.1,
    )
    early_stopping.on_train_start()
    early_stopping.on_validation_start(metrics={"mse_overfit": MSE, "psnr": PSNR})

    state.should_stop = False
    state.current_epoch = 0
    early_stopping.on_validation_end(state=state, metrics_df=METRICS)
    assert not state.should_stop

    state.current_epoch = 2
    early_stopping.on_validation_end(state=state, metrics_df=METRICS)
    assert not state.should_stop

    state.current_epoch = 4
    early_stopping.on_validation_end(state=state, metrics_df=METRICS)
    assert not state.should_stop

    state.current_epoch = 6
    early_stopping.on_validation_end(state=state, metrics_df=METRICS)
    assert not state.should_stop

    state.current_epoch = 9
    early_stopping.on_validation_end(state=state, metrics_df=METRICS)
    assert state.should_stop


def test_from_dict_to_dict():
    early_stopping = EarlyStoppingCallback(
        metric=["psnr", "mse"], patience=[2, 1], min_delta=0.1
    )
    assert isinstance(
        new_early_stopping := EarlyStoppingCallback.from_dict(early_stopping.to_dict()),
        EarlyStoppingCallback,
    )
    assert new_early_stopping.config.stoppers[0].metric == "psnr"
    assert new_early_stopping.config.stoppers[1].metric == "mse"
    assert new_early_stopping.config.stoppers[0].patience == 2
    assert new_early_stopping.config.stoppers[1].patience == 1
    assert new_early_stopping.config.stoppers[0].min_delta == 0.1
    assert new_early_stopping.config.stoppers[1].min_delta == 0.1

    early_stopping.on_train_start()
    early_stopping.on_validation_start(metrics={"psnr": PSNR, "mse": MSE})
    new_early_stopping = EarlyStoppingCallback.from_dict(early_stopping.to_dict())
    assert new_early_stopping.stoppers[0].config.metric == "psnr"
    assert new_early_stopping.stoppers[1].config.metric == "mse"
    assert new_early_stopping.stoppers[0].config.patience == 2
    assert new_early_stopping.stoppers[1].config.patience == 1
    assert new_early_stopping.stoppers[0].config.min_delta == 0.1
    assert new_early_stopping.stoppers[1].config.min_delta == 0.1
    assert new_early_stopping.stoppers[0].config.mode == "max"
    assert new_early_stopping.stoppers[1].config.mode == "min"


def test_state_dict():
    state = TrainerState()
    early_stopping = EarlyStoppingCallback(
        metric=["psnr", "mse"], patience=3, min_delta=0.1
    )
    assert early_stopping.state_dict() == dict()

    early_stopping.on_train_start()
    early_stopping.on_validation_start(metrics={"psnr": PSNR, "mse": MSE})

    state.current_epoch = 0
    early_stopping.on_validation_end(state=state, metrics_df=METRICS)

    state.current_epoch = 2
    early_stopping.on_validation_end(state=state, metrics_df=METRICS)

    state_dict = early_stopping.state_dict()

    new_early_stopping = EarlyStoppingCallback(
        metric=["psnr", "mse"], patience=3, min_delta=0.1
    )
    new_early_stopping.load_state_dict(state_dict)
    assert new_early_stopping.stoppers is None

    new_early_stopping = EarlyStoppingCallback.from_dict(early_stopping.to_dict())
    new_early_stopping.load_state_dict(state_dict)
    assert new_early_stopping.stoppers[0].best == 1.0
    assert new_early_stopping.stoppers[0].num_non_improvements == 1
    assert new_early_stopping.stoppers[1].best == -1.11
    assert new_early_stopping.stoppers[1].num_non_improvements == 0
