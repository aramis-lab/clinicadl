import builtins
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from clinicadl.callbacks.factory.monitor import PhaseMonitor, _Monitor

# -----------------------
# PhaseMonitor tests
# -----------------------


def test_phase_monitor_start_stop_summary():
    pm = PhaseMonitor()

    # start -> stop
    pm.start()
    pm.stop()

    summary = pm.summary()
    assert "time" in summary
    assert "cpu" in summary
    assert "gpu" in summary

    # stats should be non-negative
    assert summary["time"]["avg"] >= 0
    assert summary["cpu"]["avg"] >= 0
    if torch.cuda.is_available():
        assert summary["gpu"]["avg"] >= 0
    else:
        assert summary["gpu"]["avg"] == 0.0


def test_phase_monitor_stop_without_start_raises():
    pm = PhaseMonitor()
    with pytest.raises(RuntimeError):
        pm.stop()


# -----------------------
# _Monitor callback tests
# -----------------------


class FakeSplit:
    def __init__(self):
        self.train_loader = [0] * 2  # minimal loader
        self.performance_txt = Path("perf.txt")


class FakeMaps:
    def __init__(self):
        self.training = type("Training", (), {})()
        self.training.splits = [FakeSplit()]


class FakeState:
    def __init__(self):
        self.maps = FakeMaps()
        self.split = type("Split", (), {"index": 0})()
        self.epoch = 0
        self.batch = 0
        self.comp = type("Comp", (), {"device": "cpu"})()
        self.optim = type("Optim", (), {"epochs": 2})()


@pytest.fixture
def monitor():
    return _Monitor()


def test_monitor_phase_hooks(monitor):
    state = FakeState()

    # Call all hooks and check internal PhaseMonitor times are recorded
    monitor.on_train_begin(state)
    monitor.on_epoch_begin(state)
    monitor.on_batch_begin(state)
    monitor.on_backward_begin(state)
    monitor.on_backward_end(state)
    monitor.on_batch_end(state)
    monitor.on_validation_begin(state)
    monitor.on_validation_end(state)
    monitor.on_train_end(state)

    # All phases should have at least one recorded time
    for phase in [
        monitor.all_phases,
        monitor.training_phase,
        monitor.validation_phase,
        monitor.loading_phase,
        monitor.forward_phase,
        monitor.backward_phase,
    ]:
        assert len(phase.times) >= 1


@patch("builtins.open", new_callable=MagicMock)
def test_monitor_write_file(mock_open, monitor):
    state = FakeState()
    monitor.on_train_begin(state)
    monitor.on_train_end(state)

    # open() should be called with performance_txt path
    mock_open.assert_called_with(state.maps.training.splits[0].performance_txt, "w")
