import json
import shutil
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pandas as pd

from clinicadl.callbacks import TrainingCheckpointCallback
from clinicadl.io import Maps

MAPS_PATH = Path(__file__).parents[2] / "resources" / "maps_example"


def _save_files(*files: Path, **kwargs: Path) -> None:
    def _save_file(f: Path) -> None:
        if f.suffix == ".tsv":
            pd.DataFrame({"abc": [0]}).to_csv(f, sep="\t", index=False)
        else:
            with open(f, "w") as f_:
                json.dump(f.stem, f_)

    for f in files + tuple(kwargs.values()):
        _save_file(f)


STATE = Mock()
STATE.to_json.side_effect = _save_files
STATE.called = "train"
STATE.split_idx = 0

CALLBACKS = Mock()
CALLBACKS.all_callbacks = [Mock(), Mock(), MagicMock()]
for c in CALLBACKS.all_callbacks[:2]:
    c.state_dict.return_value = type(c).__name__
CALLBACKS.all_callbacks[2].state_dict.return_value = None

METRICS = Mock()
METRICS.save.side_effect = _save_files

OPTIMIZERS = {"adam": Mock(), "sgd": Mock()}
for name, opt in OPTIMIZERS.items():
    opt.state_dict.return_value = name

SCALER = Mock()
SCALER.state_dict.return_value = "scaler"

MODEL = Mock()
MODEL.state_dict.return_value = "model"


def test_disabled(caplog, tmp_path):
    shutil.copytree(MAPS_PATH, tmp_path, dirs_exist_ok=True)
    maps = Maps(tmp_path)
    maps.read()
    state = Mock()
    state.split_idx = 2
    maps.training.create_split(state.split_idx)
    state.current_epoch = 1

    chkpt = TrainingCheckpointCallback(every_n_epochs=1, enabled=False)
    with caplog.at_level("DEBUG"):
        chkpt.on_train_start(
            callbacks=CALLBACKS,
            metrics=METRICS,
            optimizers=OPTIMIZERS,
            grad_scaler=SCALER,
        )
        chkpt.on_exception(maps=maps, state=state)
        chkpt.on_epoch_end(state=state, model=MODEL, maps=maps)
        assert not (maps.training.splits[state.split_idx].tmp.path).exists()
        chkpt.on_train_end(maps=maps, state=state)
    assert len(caplog.records) == 0


def test_saving(caplog, tmp_path):
    shutil.copytree(MAPS_PATH, tmp_path, dirs_exist_ok=True)
    maps = Maps(tmp_path)
    maps.read()

    chkpt = TrainingCheckpointCallback(every_n_epochs=2)

    tmp_dir = maps.training.splits[0].tmp
    tmp_dir.clear()

    chkpt.on_train_start(
        callbacks=CALLBACKS, metrics=METRICS, optimizers=OPTIMIZERS, grad_scaler=SCALER
    )

    with caplog.at_level("INFO"):
        chkpt.on_exception(maps=maps, state=STATE)
    assert len(caplog.records) == 0

    for epoch in range(1, 6):
        STATE.current_epoch = epoch
        if epoch == 5:
            with caplog.at_level("INFO"):
                chkpt.on_exception(maps=maps, state=STATE)
            break
        with caplog.at_level("DEBUG"):
            chkpt.on_epoch_end(state=STATE, model=MODEL, maps=maps)

        if epoch == 1:
            assert tmp_dir.epochs_list == []
        elif epoch == 2:
            assert tmp_dir.epochs_list == [2]
            assert "Training checkpoint saved after epoch 2" in caplog.text
        elif epoch == 3:
            assert tmp_dir.epochs_list == [2]

    assert "Last checkpoint at the end of epoch 4" in caplog.text

    assert tmp_dir.epochs_list == [4]
    assert maps.open_file(maps.training.splits[0].tmp.epochs[4].state_json) == "state"
    assert maps.open_file(maps.training.splits[0].tmp.epochs[4].model_pt) == "model"
    assert maps.open_file(maps.training.splits[0].tmp.epochs[4].optimizer_pt) == {
        "adam": "adam",
        "sgd": "sgd",
    }
    assert maps.open_file(maps.training.splits[0].tmp.epochs[4].scaler_pt) == "scaler"
    assert (
        maps.open_file(maps.training.splits[0].tmp.epochs[4].callbacks / "mock.pt")
        == "Mock"
    )
    assert (
        maps.open_file(maps.training.splits[0].tmp.epochs[4].callbacks / "mock_1.pt")
        == "Mock"
    )
    assert (
        maps.open_file(
            maps.training.splits[0].tmp.epochs[4].callbacks / "magic_mock.pt"
        )
        is None
    )
    pd.testing.assert_frame_equal(
        maps.open_file(
            maps.training.splits[0].tmp.epochs[4].validation_metrics.aggregated_tsv
        ),
        pd.DataFrame({"abc": [0]}),
    )
    pd.testing.assert_frame_equal(
        maps.open_file(
            maps.training.splits[0].tmp.epochs[4].validation_metrics.details_tsv
        ),
        pd.DataFrame({"abc": [0]}),
    )

    chkpt.on_train_end(maps=maps, state=STATE)
    assert tmp_dir.epochs_list == []


def test_resume(caplog):
    maps = Maps(MAPS_PATH)
    maps.read()
    chkpt = TrainingCheckpointCallback()

    metrics = Mock()
    metrics.metrics = {"metric1": Mock(), "metric2": Mock()}

    with caplog.at_level("INFO"), patch.object(
        TrainingCheckpointCallback, "on_train_start"
    ) as on_train_start:
        chkpt.on_resume(
            state=STATE,
            model=MODEL,
            maps=maps,
            metrics=metrics,
            callbacks=CALLBACKS,
            optimizers=OPTIMIZERS,
            grad_scaler=SCALER,
        )
    assert "Loading checkpoints from epoch 3" in caplog.text
    on_train_start.assert_called_once_with(
        callbacks=CALLBACKS, metrics=metrics, optimizers=OPTIMIZERS, grad_scaler=SCALER
    )

    STATE.load_state_dict.assert_called_once_with({"current_epoch": 3})
    MODEL.load_state_dict.assert_called_once_with({"linear.0": 1.0})
    OPTIMIZERS["adam"].load_state_dict.assert_called_once_with({"last_epoch": 3})
    OPTIMIZERS["sgd"].load_state_dict.assert_called_once_with({"last_epoch": 4})
    SCALER.load_state_dict.assert_called_once_with({"scale": 1e3})
    metrics.load.assert_called_once_with(
        maps.training.splits[0].tmp.epochs[3].validation_metrics.aggregated_tsv,
        details_path=maps.training.splits[0]
        .tmp.epochs[3]
        .validation_metrics.details_tsv,
    )
    metrics.remove_metrics.assert_called_once_with({"metric2"})
    CALLBACKS.all_callbacks[0].load_state_dict.assert_called_once_with({"state": 0})
    CALLBACKS.all_callbacks[1].load_state_dict.assert_called_once_with({"state": 1})
    CALLBACKS.all_callbacks[2].load_state_dict.assert_called_once_with({"state": 2})


def test_from_to_dict():
    monitor = TrainingCheckpointCallback(every_n_epochs=7)
    new_monitor = TrainingCheckpointCallback.from_dict(monitor.to_dict())
    assert new_monitor.config.every_n_epochs == 7
