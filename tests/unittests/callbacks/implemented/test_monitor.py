import shutil
import time
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest

from clinicadl.callbacks import MonitorCallback
from clinicadl.callbacks.implemented.monitor import _PhaseMonitor
from clinicadl.io import Maps

OPTIMIZATION = Mock()
COMPUTATIONAL = Mock()
STATE = Mock()
STATE.split_idx = 0
STATE.called = "train"
SPLIT = Mock()
SPLIT.train_loader = Mock()
SPLIT.val_loader = Mock()
SPLIT.train_loader.batch_size = 100
SPLIT.val_loader.batch_size = 10

MAPS_PATH = Path(__file__).parents[2] / "resources" / "maps_example"


def _check_running(*args) -> None:
    for arg in args:
        print(arg.name)
        assert arg.running


def _check_not_running(*args) -> None:
    for arg in args:
        assert not arg.running


def _training(
    monitor: MonitorCallback,
    maps: Maps,
    epochs: int = 3,
    train_batches: int = 4,
    val_batches: int = 2,
    check_running: bool = False,
    raise_error: bool = False,
    sleep_after_training_start: bool = False,
):
    if not check_running:
        _assert_running = lambda *x: True  # noqa: E731
        _assert_not_running = lambda *x: True  # noqa: E731
    else:
        _assert_running = _check_running
        _assert_not_running = _check_not_running

    STATE.num_train_batches = train_batches
    STATE.num_val_batches = val_batches

    monitor.on_train_start(
        split=SPLIT, optimization=OPTIMIZATION, computational=COMPUTATIONAL
    )

    if sleep_after_training_start:
        time.sleep(1)

    _assert_running(monitor.monitor_global_training)
    STATE.called = "train"
    for e in range(1, epochs + 1):
        STATE.stage = "training"
        STATE.current_epoch = e
        monitor.on_epoch_start()
        _assert_running(
            monitor.monitor_global_training,
            monitor.monitor_epoch,
            monitor.monitor_train_batch_loading,
        )
        for b in range(1, train_batches + 1):
            STATE.current_train_batch = b
            _assert_running(
                monitor.monitor_global_training,
                monitor.monitor_epoch,
                monitor.monitor_train_batch_loading,
            )
            monitor.on_forward_step_start()
            _assert_not_running(monitor.monitor_train_batch_loading)
            _assert_running(
                monitor.monitor_global_training,
                monitor.monitor_epoch,
                monitor.monitor_forward,
            )
            monitor.on_backward_step_start()
            _assert_not_running(monitor.monitor_forward)
            _assert_running(
                monitor.monitor_global_training,
                monitor.monitor_epoch,
                monitor.monitor_backward,
            )
            if raise_error and e > 1 and b > 1:
                raise RuntimeError("CUDA out of memory")
            monitor.on_backward_step_end()
            _assert_not_running(monitor.monitor_backward)
            _assert_running(monitor.monitor_global_training, monitor.monitor_epoch)
            if STATE.current_train_batch % OPTIMIZATION.accumulation_steps == 0:
                monitor.on_optimization_step_start()
                _assert_running(
                    monitor.monitor_global_training,
                    monitor.monitor_epoch,
                    monitor.monitor_train_loop,
                    monitor.monitor_optimization,
                )
                monitor.on_optimization_step_end()
                _assert_not_running(monitor.monitor_optimization)
                _assert_running(
                    monitor.monitor_global_training,
                    monitor.monitor_epoch,
                    monitor.monitor_train_loop,
                )
            else:
                _assert_not_running(monitor.monitor_train_loop)
            monitor.on_batch_end(state=STATE)

        _assert_not_running(
            monitor.monitor_train_loop, monitor.monitor_train_batch_loading
        )
        _assert_running(monitor.monitor_global_training, monitor.monitor_epoch)

        if (STATE.current_epoch - 1) % OPTIMIZATION.evaluation_steps == 0:
            monitor.on_validation_start()
            STATE.stage = "evaluation"
            _assert_running(
                monitor.monitor_global_training,
                monitor.monitor_epoch,
                monitor.monitor_validation,
                monitor.monitor_val_batch,
                monitor.monitor_val_batch_loading,
            )
            for b in range(1, val_batches + 1):
                STATE.current_val_batch = b
                _assert_running(
                    monitor.monitor_global_training,
                    monitor.monitor_epoch,
                    monitor.monitor_validation,
                    monitor.monitor_val_batch,
                    monitor.monitor_val_batch_loading,
                )
                monitor.on_evaluation_step_start(state=STATE)
                _assert_not_running(monitor.monitor_val_batch_loading)
                _assert_running(
                    monitor.monitor_global_training,
                    monitor.monitor_epoch,
                    monitor.monitor_validation,
                    monitor.monitor_val_batch,
                    monitor.monitor_evaluation,
                )
                monitor.on_evaluation_step_end(state=STATE)
                _assert_not_running(monitor.monitor_evaluation)
                _assert_running(
                    monitor.monitor_global_training,
                    monitor.monitor_epoch,
                    monitor.monitor_validation,
                    monitor.monitor_val_batch,
                )
                monitor.on_batch_end(state=STATE)

            _assert_not_running(
                monitor.monitor_val_batch, monitor.monitor_val_batch_loading
            )
            _assert_running(
                monitor.monitor_global_training,
                monitor.monitor_epoch,
                monitor.monitor_validation,
            )

            monitor.on_validation_end(state=STATE)

            _assert_not_running(monitor.monitor_validation)
            _assert_running(monitor.monitor_global_training, monitor.monitor_epoch)

        monitor.on_epoch_end()

        _assert_not_running(monitor.monitor_epoch)
        _assert_running(monitor.monitor_global_training)

    monitor.on_train_end(maps=maps, state=STATE)

    _assert_not_running(monitor.monitor_global_training)


def test_monitor(tmp_path):
    shutil.copytree(MAPS_PATH, tmp_path, dirs_exist_ok=True)
    maps = Maps(tmp_path)
    maps.read()

    OPTIMIZATION.accumulation_steps = 2
    OPTIMIZATION.evaluation_steps = 2
    COMPUTATIONAL.gpu = False

    monitor = MonitorCallback(num_measurements=100, warmup_iterations=5)

    _training(monitor, maps)

    df = pd.read_csv(
        maps.training.splits[STATE.split_idx].logs.computational_tsv, sep="\t"
    )
    assert len(df["measurement #"].dropna()) == 2 * 4
    assert len(df["Training (s)"].dropna()) == 1
    assert len(df["Training (s)"].dropna()) == 1
    assert len(df["Epoch (s)"].dropna()) == 3
    assert len(df["Training loop (s)"].dropna()) == 2 * 2
    assert len(df["Training data loading (s)"].dropna()) == 2 * 4
    assert len(df["Forward (s)"].dropna()) == 2 * 4
    assert len(df["Forward GPU (s)"].dropna()) == 0
    assert len(df["Forward GPU max memory (MB)"].dropna()) == 0
    assert len(df["Backward (s)"].dropna()) == 2 * 4
    assert len(df["Backward GPU (s)"].dropna()) == 0
    assert len(df["Backward GPU max memory (MB)"].dropna()) == 0
    assert len(df["Optimization (s)"].dropna()) == 2 * 2
    assert len(df["Optimization GPU (s)"].dropna()) == 0
    assert len(df["Optimization GPU max memory (MB)"].dropna()) == 0
    assert len(df["Validation (s)"].dropna()) == 2
    assert len(df["Validation loop (s)"].dropna()) == 1 + 2
    assert len(df["Validation data loading (s)"].dropna()) == 1 + 2
    assert len(df["Evaluation (s)"].dropna()) == 1 + 2
    assert len(df["Evaluation GPU (s)"].dropna()) == 0
    assert len(df["Evaluation GPU max memory (MB)"].dropna()) == 0

    summary = maps.open_file(maps.training.splits[STATE.split_idx].summary_log)
    print(summary)
    assert "Training completed after 1,000 epochs\n\n***" in summary
    assert "GPU:" not in summary
    assert "GPU throughput:" not in summary

    # check running
    monitor = MonitorCallback(num_measurements=100, warmup_iterations=0)
    _training(monitor, maps, check_running=True)

    # disable
    monitor = MonitorCallback(warmup_iterations=0, enabled=False)
    _training(monitor, maps)
    df = pd.read_csv(
        maps.training.splits[STATE.split_idx].logs.computational_tsv, sep="\t"
    )
    assert len(df) == 0

    # measurement limit
    monitor = MonitorCallback(num_measurements=5, warmup_iterations=5)
    _training(monitor, maps)
    df = pd.read_csv(
        maps.training.splits[STATE.split_idx].logs.computational_tsv, sep="\t"
    )
    assert len(df["Training loop (s)"].dropna()) == 2 * 2
    assert len(df["Training data loading (s)"].dropna()) == 5


def test_exception(caplog, tmp_path):
    shutil.copytree(MAPS_PATH, tmp_path, dirs_exist_ok=True)
    maps = Maps(tmp_path)
    maps.read()

    OPTIMIZATION.accumulation_steps = 1
    OPTIMIZATION.evaluation_steps = 1
    COMPUTATIONAL.gpu = False

    monitor = MonitorCallback(warmup_iterations=0)

    try:
        _training(
            monitor,
            maps,
            raise_error=True,
            epochs=2,
            train_batches=2,
            val_batches=2,
            sleep_after_training_start=True,
        )
    except Exception as e:
        with caplog.at_level("ERROR"):
            monitor.on_exception(maps=maps, state=STATE, exception=e)

    df = pd.read_csv(
        maps.training.splits[STATE.split_idx].logs.computational_tsv, sep="\t"
    )
    assert len(df) == 4
    assert (
        f"CUDA out of memory. To debug, you can have a look at your memory usage in {maps.training.splits[STATE.split_idx].logs.computational_tsv}"
        in caplog.text
    )


def test_checkpoint(tmp_path):
    shutil.copytree(MAPS_PATH, tmp_path, dirs_exist_ok=True)
    maps = Maps(tmp_path)
    maps.read()

    OPTIMIZATION.accumulation_steps = 1
    OPTIMIZATION.evaluation_steps = 1
    COMPUTATIONAL.gpu = False

    monitor = MonitorCallback(warmup_iterations=0)

    with pytest.raises(RuntimeError):
        _training(
            monitor,
            maps,
            raise_error=True,
            epochs=2,
            train_batches=2,
            val_batches=2,
            sleep_after_training_start=True,
        )

    state_dict = monitor.state_dict()
    assert state_dict["Training"]["elapsed"] > 1
    monitor = MonitorCallback(warmup_iterations=0)
    monitor.on_resume(
        split=SPLIT, optimization=OPTIMIZATION, computational=COMPUTATIONAL
    )
    monitor.load_state_dict(state_dict)

    assert monitor.n_iterations == 0
    assert len(monitor.monitor_global_training.times) == 0
    assert len(monitor.monitor_epoch.times) == 1
    assert len(monitor.monitor_train_loop.times) == 3
    assert len(monitor.monitor_train_batch_loading.times) == 4
    assert len(monitor.monitor_forward.times) == 4
    assert len(monitor.monitor_backward.times) == 3
    assert len(monitor.monitor_optimization.times) == 3
    assert len(monitor.monitor_evaluation.times) == 2
    assert len(monitor.monitor_val_batch.times) == 2
    assert len(monitor.monitor_val_batch_loading.times) == 2
    assert len(monitor.monitor_validation.times) == 1

    monitor.on_train_end(maps=maps, state=STATE)
    assert monitor.monitor_global_training.times[0] > 1


def test_from_to_dict():
    monitor = MonitorCallback(num_measurements=100, warmup_iterations=5, enabled=False)
    new_monitor = MonitorCallback.from_dict(monitor.to_dict())
    assert new_monitor.config.num_measurements == 100
    assert new_monitor.config.warmup_iterations == 5
    assert not new_monitor.config.enabled


@pytest.mark.gpu
def test_monitor_gpu(tmp_path):
    shutil.copytree(MAPS_PATH, tmp_path, dirs_exist_ok=True)
    maps = Maps(tmp_path)
    maps.read()

    OPTIMIZATION.accumulation_steps = 2
    OPTIMIZATION.evaluation_steps = 2
    COMPUTATIONAL.gpu = True

    monitor = MonitorCallback(num_measurements=100, warmup_iterations=5)

    _training(monitor, maps)
    df = pd.read_csv(
        maps.training.splits[STATE.split_idx].logs.computational_tsv, sep="\t"
    )
    assert len(df["measurement #"].dropna()) == 2 * 4
    assert len(df["Forward GPU (s)"].dropna()) == 2 * 4
    assert len(df["Forward GPU max memory (MB)"].dropna()) == 2 * 4
    assert len(df["Backward GPU (s)"].dropna()) == 2 * 4
    assert len(df["Backward GPU max memory (MB)"].dropna()) == 2 * 4
    assert len(df["Optimization GPU (s)"].dropna()) == 2 * 2
    assert len(df["Optimization GPU max memory (MB)"].dropna()) == 2 * 2
    assert len(df["Evaluation GPU (s)"].dropna()) == 1 + 2
    assert len(df["Evaluation GPU max memory (MB)"].dropna()) == 1 + 2

    summary = maps.open_file(maps.training.splits[STATE.split_idx].summary_log)

    assert "GPU:" in summary
    assert "GPU throughput:" in summary

    # multiple gpus
    monitor = MonitorCallback(warmup_iterations=0)

    with pytest.raises(RuntimeError):
        _training(
            monitor,
            maps,
            raise_error=True,
            epochs=2,
            train_batches=2,
            val_batches=2,
            sleep_after_training_start=True,
        )

    state_dict = monitor.state_dict()
    monitor = MonitorCallback(warmup_iterations=0)
    monitor.on_train_start(
        split=SPLIT, optimization=OPTIMIZATION, computational=COMPUTATIONAL
    )
    monitor.load_state_dict(state_dict)
    assert len(monitor._gpus_used) == 2


def test_phase_monitor():
    monitor = _PhaseMonitor(gpu=False, num_measurements=2)
    monitor.start()
    time.sleep(0.1)
    monitor.stop()
    monitor.start()
    assert "elapsed" not in monitor.state_dict()
    monitor.stop()
    monitor.start()
    monitor.stop()
    assert len(monitor.times) == 2
    assert 0.09 <= monitor.times[0]

    monitor = _PhaseMonitor(gpu=False, num_measurements=1, save_time=True)
    monitor.start()
    time.sleep(0.1)
    state_dict = monitor.state_dict()
    monitor = _PhaseMonitor(gpu=False, num_measurements=1, save_time=True)
    monitor.start()
    monitor.load_state_dict(state_dict)
    monitor.stop()
    assert 0.09 <= state_dict["elapsed"]
    assert monitor.times[0] >= 0.05

    monitor = _PhaseMonitor(gpu=False, num_measurements=1, enabled=False)
    monitor.start()
    monitor.stop()
    assert len(monitor.times) == 0


@pytest.mark.gpu
def test_phase_monitor_gpu():
    import torch

    x = torch.randn(2048, 2048, device="cuda")
    monitor = _PhaseMonitor(gpu=True, num_measurements=1)
    monitor.start()
    _ = x @ x
    monitor.stop()
    assert monitor.gpu_times[0] >= 1
    assert monitor.gpu_max_mem[0] >= 1e7

    monitor = _PhaseMonitor(gpu=True, num_measurements=1, memory=False)
    monitor.start()
    monitor.stop()
    assert monitor.gpu_max_mem == []
