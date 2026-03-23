import shutil
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

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
    MetricsSaverCallback,
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
    assert cb_handler._first_and_last[0] is log

    assert isinstance(cb_handler._ordered[0], ChecksCallback)
    assert isinstance(cb_handler._ordered[1], ConfigSaverCallback)
    assert isinstance(cb_handler._ordered[2], MonitorCallback)
    assert cb_handler._ordered[2].config.num_measurements == 100
    assert cb_handler._ordered[3] is es_1
    assert cb_handler._ordered[4] is es_2
    assert cb_handler._ordered[5] is custom
    assert isinstance(cb_handler._ordered[6], ModelCheckpointCallback)
    assert cb_handler._ordered[6].config.save_last
    assert isinstance(cb_handler._ordered[7], TrainingLossCallback)
    assert isinstance(cb_handler._ordered[8], MetricsSaverCallback)
    assert cb_handler._ordered[9] is chkpt
    assert len(cb_handler._ordered) == 10

    assert cb_handler.callbacks[0] is log
    assert isinstance(cb_handler.callbacks[1], MonitorCallback)
    assert cb_handler.callbacks[2] is es_1
    assert cb_handler.callbacks[3] is es_2
    assert cb_handler.callbacks[4] is custom
    assert isinstance(cb_handler.callbacks[5], ModelCheckpointCallback)
    assert cb_handler.callbacks[6] is chkpt
    assert len(cb_handler.callbacks) == 7

    assert len(cb_handler.all_callbacks) == 11
    assert cb_handler.all_callbacks[:7] == cb_handler.callbacks
    assert isinstance(cb_handler.all_callbacks[7], ChecksCallback)
    assert isinstance(cb_handler.all_callbacks[8], ConfigSaverCallback)
    assert isinstance(cb_handler.all_callbacks[9], TrainingLossCallback)
    assert isinstance(cb_handler.all_callbacks[10], MetricsSaverCallback)

    for cb in [LoggerCallback(), MonitorCallback(), TrainingCheckpointCallback()]:
        with pytest.raises(
            ValueError, match=f"You cannot pass more than one {type(cb)}"
        ):
            CallbacksHandler([cb, cb])

    cb_handler = CallbacksHandler([])
    assert len(cb_handler._first_and_last) == 1
    assert isinstance(cb_handler._first_and_last[0], LoggerCallback)
    cb_handler.add_callbacks(
        [
            log := LoggerCallback(debug=False),
        ]
    )
    assert cb_handler._first_and_last[0] is log
    assert cb_handler.callbacks[0] is log


def test_call_event():
    mandatory_cb = Mock()

    log = LoggerCallback()
    log.on_trainer_init = Mock()
    log.on_forward_step_start = Mock()
    log.on_resume = Mock()
    log.on_train_end = Mock()
    log.on_exception = Mock()
    monitor = MonitorCallback()
    monitor.on_trainer_init = Mock()
    monitor.on_forward_step_start = Mock()
    monitor.on_resume = Mock()
    monitor.on_train_end = Mock()
    monitor.on_exception = Mock()
    cb = Mock()
    cb.__class__ = Callback

    on_trainer_init = Mock()
    on_trainer_init.attach_mock(log.on_trainer_init, "log")
    on_trainer_init.attach_mock(monitor.on_trainer_init, "monitor")
    on_trainer_init.attach_mock(cb.on_trainer_init, "cb")
    on_trainer_init.attach_mock(mandatory_cb.on_trainer_init, "mandatory")
    on_forward_step_start = Mock()
    on_forward_step_start.attach_mock(log.on_forward_step_start, "log")
    on_forward_step_start.attach_mock(monitor.on_forward_step_start, "monitor")
    on_forward_step_start.attach_mock(cb.on_forward_step_start, "cb")
    on_forward_step_start.attach_mock(mandatory_cb.on_forward_step_start, "mandatory")
    on_resume = Mock()
    on_resume.attach_mock(log.on_resume, "log")
    on_resume.attach_mock(monitor.on_resume, "monitor")
    on_resume.attach_mock(cb.on_resume, "cb")
    on_resume.attach_mock(mandatory_cb.on_resume, "mandatory")
    on_train_end = Mock()
    on_train_end.attach_mock(log.on_train_end, "log")
    on_train_end.attach_mock(monitor.on_train_end, "monitor")
    on_train_end.attach_mock(cb.on_train_end, "cb")
    on_train_end.attach_mock(mandatory_cb.on_train_end, "mandatory")
    on_exception = Mock()
    on_exception.attach_mock(log.on_exception, "log")
    on_exception.attach_mock(monitor.on_exception, "monitor")
    on_exception.attach_mock(cb.on_exception, "cb")
    on_exception.attach_mock(mandatory_cb.on_exception, "mandatory")

    with patch(
        "clinicadl.callbacks.handler.CallbacksHandler._get_mandatory"
    ) as mock_mandatory, patch(
        "clinicadl.callbacks.handler.CallbacksHandler._get_default"
    ) as mock_default:
        mock_mandatory.return_value = [mandatory_cb]
        mock_default.return_value = []

        cb_handler = CallbacksHandler([log, Callback(), monitor, cb])

    with pytest.raises(TypeError, match="missing 5 required keyword-only argument"):
        cb_handler.call_event(
            "on_trainer_init",
            model=Mock(),
        )

    on_trainer_init.reset_mock()
    log.on_trainer_init.reset_mock()
    cb_handler.call_event(
        "on_trainer_init",
        model=Mock(),
        maps=Mock(),
        state=Mock(),
        metrics=Mock(),
        optimization=Mock(),
        callbacks=Mock(),
    )
    cb_handler.call_event(
        "on_forward_step_start",
        model=Mock(),
        maps=Mock(),
        state=Mock(),
        batch=Mock(),
    )
    cb_handler.call_event(
        "on_resume",
        model=Mock(),
        maps=Mock(),
        state=Mock(),
        split=Mock(),
        optimizers=Mock(),
        grad_scaler=Mock(),
        optimization=Mock(),
        metrics=Mock(),
        callbacks=Mock(),
        computational=Mock(),
    )
    cb_handler.call_event(
        "on_train_end",
        model=Mock(),
        maps=Mock(),
        state=Mock(),
    )
    cb_handler.call_event(
        "on_exception",
        model=Mock(),
        maps=Mock(),
        state=Mock(),
        exception=Mock(),
    )
    assert [c[0] for c in on_trainer_init.mock_calls] == [
        "log",
        "monitor",
        "cb",
        "mandatory",
    ]
    assert [c[0] for c in on_forward_step_start.mock_calls] == [
        "log",
        "monitor",
        "cb",
        "mandatory",
    ]
    assert [c[0] for c in on_resume.mock_calls] == ["log", "monitor", "cb", "mandatory"]
    assert [c[0] for c in on_train_end.mock_calls] == [
        "monitor",
        "cb",
        "mandatory",
        "log",
    ]
    assert [c[0] for c in on_exception.mock_calls] == [
        "monitor",
        "cb",
        "mandatory",
        "log",
    ]


def test_checkpoints(tmp_path):
    maps = Maps(tmp_path)
    maps.training.create_split(1)

    chkpt = TrainingCheckpointCallback(every_n_epochs=1)
    cb = Mock()
    cb.__class__ = Callback
    cb.state_dict = Mock()
    cb.state_dict.return_value = "callback"
    log = Mock()
    log.state_dict = Mock()
    log.state_dict.return_value = "logger"
    loss = Mock()
    loss.state_dict = Mock()
    loss.state_dict.return_value = "loss"

    with patch(
        "clinicadl.callbacks.handler.CallbacksHandler._get_mandatory"
    ) as mock_mandatory, patch(
        "clinicadl.callbacks.handler.CallbacksHandler._get_default"
    ) as mock_default:
        mock_mandatory.return_value = [loss]
        mock_default.return_value = [log]

        cb_handler = CallbacksHandler([cb, chkpt])

    cb_handler.call_event(
        "on_train_start",
        metrics=Mock(),
        callbacks=cb_handler,
        optimizers=Mock(),
        grad_scaler=Mock(),
    )

    state = Mock()
    state.current_epoch = 1
    state.split_idx = 1
    model = Mock()
    model.state_dict.return_value = {}
    opt = {"opt": Mock()}
    opt["opt"].state_dict.return_value = {}
    scaler = Mock()
    scaler.state_dict.return_value = {}

    chkpt._metrics = Mock()
    chkpt._optimizers = opt
    chkpt._scaler = scaler

    cb_handler.call_event("on_epoch_end", model=model, maps=maps, state=state)
    assert (
        maps.open_file(
            maps.training.splits[state.split_idx].tmp.epochs[1].callbacks / "mock.pt"
        )
        == "callback"
    )
    assert (
        maps.open_file(
            maps.training.splits[state.split_idx].tmp.epochs[1].callbacks / "mock_1.pt"
        )
        == "logger"
    )
    assert (
        maps.open_file(
            maps.training.splits[state.split_idx].tmp.epochs[1].callbacks / "mock_2.pt"
        )
        == "loss"
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
    assert config["callbacks"][0]["name_"] == "EarlyStoppingCallback"
    assert config["callbacks"][1]["name_"] == "LoggerCallback"
