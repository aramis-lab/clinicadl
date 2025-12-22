from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from clinicadl.callbacks.base import Callback
from clinicadl.callbacks.handler import _CallbacksHandler
from clinicadl.callbacks.implemented.logger import _Logger
from clinicadl.callbacks.implemented.monitor import _Monitor
from clinicadl.callbacks.implemented.training_loss import _TrainingLoss
from clinicadl.io.maps.training.splits import TrainingSplitDir


# ----------------------------
# Fake TrainerState
# ----------------------------
class FakeState:
    def __init__(self):
        self.epoch = 0
        self.batch = 0
        self.split = type(
            "Split",
            (),
            {"index": 0, "train_loader": MagicMock(), "val_loader": MagicMock()},
        )()
        self.maps = type("Maps", (), {})()
        self.maps.training = type("Training", (), {})()
        self.maps.training.splits = [TrainingSplitDir(8, Path("."))]
        self.comp = type("Comp", (), {"device": "cpu"})()
        self.optim = type("Optim", (), {"epochs": 2})()


class FakeMetrics:
    def __init__(self):
        self.metrics = {"loss": None, "mae": None, "mse": None}


# ----------------------------
# Tests
# ----------------------------


def test_default_callbacks_added():
    metrics = FakeMetrics()
    handler = _CallbacksHandler(metrics=metrics, callbacks=None)

    # Check defaults
    names = handler.callback_list
    assert "_Monitor" in names
    assert "_TrainingLoss" in names
    assert "_Logger" in names


def test_callback_order():
    metrics = FakeMetrics()
    # Provide callbacks out of order
    cb1 = _Logger()
    cb2 = _Monitor()
    cb3 = _TrainingLoss()
    handler = _CallbacksHandler(metrics=metrics, callbacks=[cb1, cb2, cb3])

    ordered = handler.callback_list
    # _TrainingLoss should come first
    assert ordered[0] == "_TrainingLoss"
    assert ordered[1] == "LRScheduler" or "_Monitor" in ordered


@patch("clinicadl.callbacks.handler.write_json")
def test_write_json_calls_write_json(mock_write, tmp_path):
    metrics = FakeMetrics()
    cb = _TrainingLoss()
    handler = _CallbacksHandler(metrics=metrics, callbacks=[cb])

    json_file = tmp_path / "callbacks.json"
    handler.write_json(json_file)

    mock_write.assert_called_once()
    args, kwargs = mock_write.call_args
    assert kwargs["json_path"] == json_file
    assert isinstance(kwargs["data"], dict)


@patch("clinicadl.callbacks.handler.read_json")
@patch("clinicadl.callbacks.handler.get_callback_from_dict")
def test_from_json_recreates_callbacks(mock_get_cb, mock_read_json, tmp_path):
    json_file = tmp_path / "callbacks.json"
    mock_read_json.return_value = {"_TrainingLoss": {"name": "_TrainingLoss"}}
    mock_cb = MagicMock(spec=Callback)
    mock_get_cb.return_value = mock_cb

    callbacks = _CallbacksHandler.from_json(json_file)
    assert callbacks[0] == mock_cb
