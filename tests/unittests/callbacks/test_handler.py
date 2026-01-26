import shutil
from copy import copy
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

import clinicadl.callbacks.handler
from clinicadl.callbacks import (
    Callback,
    CallbacksHandler,
    EarlyStoppingCallback,
    LoggerCallback,
    ModelCheckpointCallback,
    MonitorCallback,
    TrainingCheckpointCallback,
)
from clinicadl.callbacks.implemented import (
    ChecksCallback,
    ConfigSaverCallback,
    TrainingLossCallback,
)
from clinicadl.io import Maps

MAPS_PATH = Path(__file__).parents[1] / "resources" / "maps_example"


class CustomCallback(Callback):
    def __init__(self):
        self.epoch = 0

    def on_epoch_end(self, *, state, **kwargs):
        self.epoch = state.current_epoch

    def state_dict(self):
        return {"abc": self.epoch}

    def load_state_dict(self, state_dict):
        pass


def test_inputs():
    cb_handler = CallbacksHandler(
        [
            es_1 := EarlyStoppingCallback(metric="loss"),
            es_2 := EarlyStoppingCallback(metric="mse"),
            chkpt := TrainingCheckpointCallback(every_n_epochs=5),
            log := LoggerCallback(debug=False),
            custom := CustomCallback(),
        ]
    )
    assert cb_handler._all_callbacks[0] is log
    assert isinstance(cb_handler._all_callbacks[1], ChecksCallback)
    assert isinstance(cb_handler._all_callbacks[2], ConfigSaverCallback)
    assert isinstance(cb_handler._all_callbacks[3], MonitorCallback)
    assert cb_handler._all_callbacks[3].config.num_measurements == 100
    assert cb_handler._all_callbacks[4] is es_1
    assert cb_handler._all_callbacks[5] is es_2
    assert cb_handler._all_callbacks[6] is custom
    assert isinstance(cb_handler._all_callbacks[7], TrainingLossCallback)
    assert isinstance(cb_handler._all_callbacks[8], ModelCheckpointCallback)
    assert cb_handler._all_callbacks[8].config.save_last
    assert cb_handler._all_callbacks[9] is chkpt
    assert len(cb_handler._all_callbacks) == 10

    assert cb_handler.callbacks[0] is log
    assert isinstance(cb_handler.callbacks[1], MonitorCallback)
    assert cb_handler.callbacks[2] is es_1
    assert cb_handler.callbacks[3] is es_2
    assert cb_handler.callbacks[4] is custom
    assert isinstance(cb_handler.callbacks[5], ModelCheckpointCallback)
    assert cb_handler.callbacks[6] is chkpt
    assert len(cb_handler.callbacks) == 7

    cb_handler.add_callbacks(
        [
            es_3 := EarlyStoppingCallback(metric="mae"),
            model_ckpt := ModelCheckpointCallback(epochs=[5]),
            monitor := MonitorCallback(num_measurements=5),
        ]
    )
    assert len(cb_handler.callbacks) == 8
    assert cb_handler._all_callbacks[3] is monitor
    assert cb_handler._all_callbacks[7] is es_3
    assert cb_handler._all_callbacks[9] is model_ckpt
    assert len(cb_handler._all_callbacks) == 11
    assert cb_handler.callbacks[1] is monitor
    assert cb_handler.callbacks[5] is es_3
    assert cb_handler.callbacks[6] is model_ckpt

    for cb in [LoggerCallback(), MonitorCallback(), TrainingCheckpointCallback()]:
        with pytest.raises(
            ValueError, match=f"You cannot pass more than one {type(cb)}"
        ):
            CallbacksHandler([cb, cb])


MANDATORY = copy(clinicadl.callbacks.handler.MANDATORY)
MANDATORY[-1] = Mock()


@patch("clinicadl.callbacks.handler.MANDATORY", MANDATORY)
def test_call_event():
    log = LoggerCallback()
    log.on_trainer_init = Mock()
    cb = Mock()
    cb.__class__ = Callback
    cb_handler = CallbacksHandler([cb, log])
    cb_handler.call_event(
        "on_trainer_init",
        model=Mock(),
        maps=Mock(),
        state=Mock(),
        metrics=Mock(),
        optimization=Mock(),
        callbacks=Mock(),
    )
    assert cb_handler.callbacks[0] is log
    log.on_trainer_init.assert_called_once()
    cb.on_trainer_init.assert_called_once()
    MANDATORY[-1].on_trainer_init.assert_called_once()


def test_checkpoints(tmp_path):
    shutil.copytree(MAPS_PATH, tmp_path, dirs_exist_ok=True)
    maps = Maps(tmp_path)
    maps.read()
    log = LoggerCallback()
    log.state_dict = Mock()
    log.state_dict.return_value = "logger"
    state = Mock()
    state.current_epoch = 1
    state.split_idx = 1
    model = Mock()
    model.state_dict.return_value = {}
    opt = {"opt": Mock()}
    opt["opt"].state_dict.return_value = {}
    scaler = Mock()
    scaler.state_dict.return_value = {}

    cb_handler = CallbacksHandler(
        [log, CustomCallback(), TrainingCheckpointCallback(every_n_epochs=1)]
    )
    # to avoid errors
    maps.callbacks_json.unlink()
    cb_handler.call_event(
        "on_trainer_init",
        state=state,
        model=model,
        maps=maps,
        metrics=Mock(),
        callbacks=cb_handler,
        optimization=Mock(),
    )
    cb_handler._all_callbacks[0].logger = Mock()
    cb_handler._all_callbacks[0]._train_progress_bar = Mock()
    cb_handler._all_callbacks[3].monitor_epoch = Mock()
    cb_handler._all_callbacks[3].monitor_epoch.state_dict.return_value = {}
    cb_handler._all_callbacks[3].monitor_epoch.name = "epoch"
    cb_handler._all_callbacks[-1]._optimizers = opt
    cb_handler._all_callbacks[-1]._scaler = scaler
    cb_handler._all_callbacks[-1]._metrics = Mock()
    #
    cb_handler.call_event("on_epoch_end", state=state, model=model, maps=maps)
    assert maps.callbacks_json.is_file()
    assert maps.open_file(
        maps.training.splits[state.split_idx].tmp.epochs[1].callbacks
        / "custom_callback.pt"
    ) == {"abc": 1}
    assert (
        maps.open_file(
            maps.training.splits[state.split_idx].tmp.epochs[1].callbacks
            / "logger_callback.pt"
        )
        == "logger"
    )


def test_serialize_deserialize(tmp_path):
    cb_handler = CallbacksHandler(
        [EarlyStoppingCallback(metric="loss"), LoggerCallback()]
    )
    cb_handler.to_json(tmp_path / "cb.json")
    config = Maps.open_file(tmp_path / "cb.json")
    cb_handler = CallbacksHandler.from_json(tmp_path / "cb.json")
    assert isinstance(cb_handler.callbacks[2], EarlyStoppingCallback)
    assert cb_handler.callbacks[2].config.stoppers[0].metric == "loss"
    assert len(config["callbacks"]) == 2
    assert config["callbacks"][0]["name"] == "EarlyStoppingCallback"
    assert config["callbacks"][1]["name"] == "LoggerCallback"
