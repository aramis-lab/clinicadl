import shutil
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from clinicadl.callbacks import ModelCheckpointCallback
from clinicadl.io import Maps
from clinicadl.metrics.config import (
    MAEMetricConfig,
    PSNRMetricConfig,
)
from clinicadl.train import TrainerState

MAE = MAEMetricConfig().get_object()
PSNR = PSNRMetricConfig(max_val=1).get_object()
MAPS_PATH = Path(__file__).parents[2] / "resources" / "maps_example"

METRICS_EPOCH_1 = pd.DataFrame(
    {
        "epoch": [1],
        "psnr": [1.0],
        "mae": [-1.0],
    }
)
DETAILED_METRICS_EPOCH_1 = pd.DataFrame(
    {
        "epoch": [1, 1],
        "participant_id": ["sub-000", "sub-001"],
        "session_id": ["ses-M000", "ses-M000"],
        "psnr": [0, 1],
        "mae": [5, 4],
    }
)
METRICS_EPOCH_2 = pd.DataFrame(
    {
        "epoch": [2],
        "psnr": [1.001],
        "mae": [-1.10],
    }
)
DETAILED_METRICS_EPOCH_2 = pd.DataFrame(
    {
        "epoch": [2, 2],
        "participant_id": ["sub-000", "sub-001"],
        "session_id": ["ses-M000", "ses-M000"],
        "psnr": [2, 3],
        "mae": [3, 2],
    }
)
METRICS_EPOCH_4 = pd.DataFrame(
    {
        "epoch": [4],
        "psnr": [1.0009],
        "mae": [np.nan],
    }
)
DETAILED_METRICS_EPOCH_4 = pd.DataFrame(
    {
        "epoch": [4, 4],
        "participant_id": ["sub-000", "sub-001"],
        "session_id": ["ses-M000", "ses-M000"],
        "psnr": [4, 5],
        "mae": [1, 0],
    }
)

METRICS = pd.concat([METRICS_EPOCH_1, METRICS_EPOCH_2, METRICS_EPOCH_4])
DETAILED_METRICS = pd.concat(
    [DETAILED_METRICS_EPOCH_1, DETAILED_METRICS_EPOCH_2, DETAILED_METRICS_EPOCH_4]
)


def compare_files(
    maps: Maps,
    model_dir,
    expected_state_dict,
    expected_metrics_df,
    expected_detailed_metrics_df,
):
    dict_ = maps.open_file(model_dir.model)
    assert dict_ == expected_state_dict

    metrics_df = maps.open_file(model_dir.validation_metrics.aggregated)
    pd.testing.assert_frame_equal(
        metrics_df,
        expected_metrics_df,
    )

    detailed_metrics_df = maps.open_file(model_dir.validation_metrics.details)
    pd.testing.assert_frame_equal(
        detailed_metrics_df,
        expected_detailed_metrics_df,
    )


def test_inputs(tmp_path):
    shutil.copytree(MAPS_PATH, tmp_path, dirs_exist_ok=True)
    maps = Maps(tmp_path)
    state = TrainerState(current_epoch=2, split_idx=1, called="train")
    maps.training.create_split(state.split_idx)

    chkpt = ModelCheckpointCallback()
    assert chkpt.config.save_last
    assert chkpt.config.epochs == []

    chkpt = ModelCheckpointCallback(save_last=False)
    assert not chkpt.config.save_last

    chkpt = ModelCheckpointCallback(metric="psnr", epochs=range(1, 3))
    assert not chkpt.config.save_last
    assert chkpt.config.metric == "psnr"
    assert chkpt.config.epochs == [1, 2]

    chkpt.on_train_start(maps=maps, state=state)
    assert (
        maps.training.splits[state.split_idx]
        .models.best_models.metrics["psnr"]
        .path.is_dir()
    )
    with pytest.raises(KeyError, match="'psnr' not found in the validation metrics!"):
        chkpt.on_validation_start(state=state, metrics={"mae": MAE})
    chkpt.on_validation_start(state=state, metrics={"psnr": PSNR, "mae": MAE})
    assert chkpt.metric_monitoring.mode == "max"

    chkpt = ModelCheckpointCallback(metric="mae", epochs=range(1, 3))
    chkpt.on_validation_start(state=state, metrics={"psnr": PSNR, "mae": MAE})
    assert chkpt.metric_monitoring.mode == "min"


def test_on_train_start(tmp_path):
    shutil.copytree(MAPS_PATH, tmp_path, dirs_exist_ok=True)
    maps = Maps(tmp_path)
    state = TrainerState(current_epoch=1, split_idx=1, called="train")
    maps.training.create_split(state.split_idx)
    model = Mock()
    model.state_dict.return_value = {}
    chkpt = ModelCheckpointCallback(metric="psnr")

    chkpt.on_train_start(maps=maps, state=state)
    chkpt.on_validation_start(state=state, metrics={"psnr": PSNR})
    chkpt.on_validation_end(
        model=model,
        maps=maps,
        state=state,
        metrics_df=METRICS,
        detailed_metrics_df=DETAILED_METRICS,
    )
    chkpt.on_train_start(maps=maps, state=state)
    assert chkpt.metric_monitoring.best == -np.inf


def test_metric(tmp_path):
    shutil.copytree(MAPS_PATH, tmp_path, dirs_exist_ok=True)
    maps = Maps(tmp_path)
    state = TrainerState(current_epoch=1, split_idx=1)
    maps.training.create_split(state.split_idx)
    model = Mock()

    expected_state_dict_1 = {"abc": [1]}
    model.state_dict.return_value = expected_state_dict_1
    chkpt = ModelCheckpointCallback(metric="psnr")
    chkpt.on_validation_end(
        model=model,
        maps=maps,
        state=state,
        metrics_df=METRICS,
        detailed_metrics_df=DETAILED_METRICS,
    )
    assert not (
        maps.training.splits[state.split_idx].models.best_models.path / "best-psnr"
    ).is_dir()

    state.called = "train"
    chkpt.on_train_start(maps=maps, state=state)
    metric_dir = maps.training.splits[state.split_idx].models.best_models.metrics[
        "psnr"
    ]
    chkpt.on_validation_start(state=state, metrics={"psnr": PSNR})
    chkpt.on_validation_end(
        model=model,
        maps=maps,
        state=state,
        metrics_df=METRICS,
        detailed_metrics_df=DETAILED_METRICS,
    )
    compare_files(
        maps,
        metric_dir,
        expected_state_dict_1,
        METRICS_EPOCH_1,
        DETAILED_METRICS_EPOCH_1,
    )

    expected_state_dict_2 = {"abc": [2]}
    model.state_dict.return_value = expected_state_dict_2
    state.current_epoch = 2
    chkpt.on_validation_end(
        model=model,
        maps=maps,
        state=state,
        metrics_df=METRICS,
        detailed_metrics_df=DETAILED_METRICS,
    )
    compare_files(
        maps,
        metric_dir,
        expected_state_dict_2,
        METRICS_EPOCH_2,
        DETAILED_METRICS_EPOCH_2,
    )

    state.current_epoch = 4
    model.state_dict.return_value = {"abc": [4]}
    chkpt.on_validation_end(
        model=model,
        maps=maps,
        state=state,
        metrics_df=METRICS,
        detailed_metrics_df=DETAILED_METRICS,
    )
    compare_files(
        maps,
        metric_dir,
        expected_state_dict_2,
        METRICS_EPOCH_2,
        DETAILED_METRICS_EPOCH_2,
    )

    # with mode = "min"
    chkpt = ModelCheckpointCallback(metric="mae")
    chkpt.on_train_start(maps=maps, state=state)
    chkpt.on_validation_start(state=state, metrics={"mae": MAE})
    metric_dir = maps.training.splits[state.split_idx].models.best_models.metrics["mae"]

    model.state_dict.return_value = expected_state_dict_1
    state.current_epoch = 1
    chkpt.on_validation_end(
        model=model,
        maps=maps,
        state=state,
        metrics_df=METRICS,
        detailed_metrics_df=DETAILED_METRICS,
    )
    compare_files(
        maps,
        metric_dir,
        expected_state_dict_1,
        METRICS_EPOCH_1,
        DETAILED_METRICS_EPOCH_1,
    )

    model.state_dict.return_value = expected_state_dict_2
    state.current_epoch = 2
    chkpt.on_validation_end(
        model=model,
        maps=maps,
        state=state,
        metrics_df=METRICS,
        detailed_metrics_df=DETAILED_METRICS,
    )
    compare_files(
        maps,
        metric_dir,
        expected_state_dict_2,
        METRICS_EPOCH_2,
        DETAILED_METRICS_EPOCH_2,
    )

    model.state_dict.return_value = {"abc": [4]}
    state.current_epoch = 4
    chkpt.on_validation_end(
        model=model,
        maps=maps,
        state=state,
        metrics_df=METRICS,
        detailed_metrics_df=DETAILED_METRICS,
    )
    compare_files(
        maps,
        metric_dir,
        expected_state_dict_2,
        METRICS_EPOCH_2,
        DETAILED_METRICS_EPOCH_2,
    )


def test_epochs(tmp_path):
    shutil.copytree(MAPS_PATH, tmp_path, dirs_exist_ok=True)
    maps = Maps(tmp_path)
    state = TrainerState(current_epoch=1, split_idx=1, called="train")
    maps.training.create_split(state.split_idx)
    model = Mock()
    chkpt = ModelCheckpointCallback(epochs=[1, 2])

    expected_state_dict_1 = {"abc": [1]}
    model.state_dict.return_value = expected_state_dict_1

    chkpt.on_validation_end(
        model=model,
        maps=maps,
        state=state,
        metrics_df=METRICS,
        detailed_metrics_df=DETAILED_METRICS,
    )
    chkpt.on_epoch_end(model=model, maps=maps, state=state)

    state.current_epoch = 2
    expected_state_dict_2 = {"abc": [2]}
    model.state_dict.return_value = expected_state_dict_2
    chkpt.on_validation_end(
        model=model,
        maps=maps,
        state=state,
        metrics_df=METRICS,
        detailed_metrics_df=DETAILED_METRICS,
    )
    chkpt.on_epoch_end(model=model, maps=maps, state=state)

    state.current_epoch = 4
    expected_state_dict_4 = {"abc": [4]}
    model.state_dict.return_value = expected_state_dict_4
    chkpt.on_validation_end(
        model=model,
        maps=maps,
        state=state,
        metrics_df=METRICS,
        detailed_metrics_df=DETAILED_METRICS,
    )
    chkpt.on_epoch_end(model=model, maps=maps, state=state)

    assert (
        maps.training.splits[state.split_idx].models.checkpoints.path / "epoch-1"
    ).is_dir()
    assert (
        maps.training.splits[state.split_idx].models.checkpoints.path / "epoch-2"
    ).is_dir()
    assert not (
        maps.training.splits[state.split_idx].models.checkpoints.path / "epoch-4"
    ).is_dir()

    compare_files(
        maps,
        maps.training.splits[state.split_idx].models.checkpoints.epochs[1],
        expected_state_dict_1,
        METRICS_EPOCH_1,
        DETAILED_METRICS_EPOCH_1,
    )
    compare_files(
        maps,
        maps.training.splits[state.split_idx].models.checkpoints.epochs[2],
        expected_state_dict_2,
        METRICS_EPOCH_2,
        DETAILED_METRICS_EPOCH_2,
    )


def test_save_last(tmp_path):
    shutil.copytree(MAPS_PATH, tmp_path, dirs_exist_ok=True)
    maps = Maps(tmp_path)
    state = TrainerState(current_epoch=4, split_idx=1, called="train")
    maps.training.create_split(state.split_idx)
    model = Mock()
    chkpt = ModelCheckpointCallback(save_last=True)

    expected_state_dict_4 = {"abc": [4]}
    model.state_dict.return_value = expected_state_dict_4

    chkpt.on_validation_end(
        model=model,
        maps=maps,
        state=state,
        metrics_df=METRICS,
        detailed_metrics_df=DETAILED_METRICS,
    )
    chkpt.on_train_end(model=model, maps=maps, state=state)

    compare_files(
        maps,
        maps.training.splits[state.split_idx].models.final,
        expected_state_dict_4,
        METRICS_EPOCH_4,
        DETAILED_METRICS_EPOCH_4,
    )


def test_from_dict_to_dict(tmp_path):
    shutil.copytree(MAPS_PATH, tmp_path, dirs_exist_ok=True)
    maps = Maps(tmp_path)
    state = TrainerState(current_epoch=4, split_idx=1, called="train")
    maps.training.create_split(state.split_idx)

    chkpt = ModelCheckpointCallback(metric="psnr", epochs=[1, 2], save_last=False)
    assert isinstance(
        new_chkpt := ModelCheckpointCallback.from_dict(chkpt.to_dict()),
        ModelCheckpointCallback,
    )
    assert new_chkpt.config.metric == "psnr"
    assert new_chkpt.config.epochs == [1, 2]
    assert not new_chkpt.config.save_last

    chkpt.on_train_start(maps=maps, state=state)
    chkpt.on_validation_start(state=state, metrics={"psnr": PSNR})

    new_chkpt = ModelCheckpointCallback.from_dict(chkpt.to_dict())
    assert new_chkpt.metric_monitoring.name == "psnr"
    assert new_chkpt.metric_monitoring.mode == "max"


def test_state_dict(tmp_path):
    shutil.copytree(MAPS_PATH, tmp_path, dirs_exist_ok=True)
    state = TrainerState(current_epoch=1, split_idx=1, called="train")
    maps = Maps(tmp_path)
    maps.training.create_split(state.split_idx)
    model = Mock()
    model.state_dict.return_value = {}

    chkpt = ModelCheckpointCallback(epochs=[1, 2])
    assert (state_dict := chkpt.state_dict()) == dict()
    chkpt.load_state_dict(state_dict)

    chkpt = ModelCheckpointCallback(metric="psnr")
    chkpt.on_train_start(maps=maps, state=state)
    assert (state_dict := chkpt.state_dict()) == dict()
    chkpt.load_state_dict(state_dict)

    chkpt.on_validation_start(state=state, metrics={"psnr": PSNR})
    chkpt.on_validation_end(
        model=model,
        maps=maps,
        state=state,
        metrics_df=METRICS,
        detailed_metrics_df=DETAILED_METRICS,
    )
    state_dict = chkpt.state_dict()

    new_chkpt = ModelCheckpointCallback(metric="psnr")
    new_chkpt.load_state_dict(state_dict)
    assert new_chkpt.metric_monitoring is None

    new_chkpt = ModelCheckpointCallback.from_dict(chkpt.to_dict())
    new_chkpt.load_state_dict(state_dict)
    assert new_chkpt.metric_monitoring.best == 1.0
    assert new_chkpt.metric_monitoring.num_non_improvements == 0
