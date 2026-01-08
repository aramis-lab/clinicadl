import contextlib
import io
import logging
import shutil
import sys
import time
from pathlib import Path
from unittest.mock import Mock

import torch

from clinicadl.callbacks import LoggerCallback
from clinicadl.io import Maps
from clinicadl.train import ComputationalConfig, TrainerState

MAPS_PATH = Path(__file__).parents[2] / "resources" / "maps_example"
MODEL = Mock()


def capture_stdout(f, **kwargs):
    stdout_capture = io.StringIO()
    original_stdout = sys.stdout
    try:
        sys.stdout = stdout_capture
        f(**kwargs)
    finally:
        sys.stdout = original_stdout

    return stdout_capture.getvalue()


def test_train(caplog, tmp_path):
    shutil.copytree(MAPS_PATH, tmp_path, dirs_exist_ok=True)
    maps = Maps(tmp_path)
    maps.exec.remove(non_empty_ok=True)
    state = TrainerState(
        split_idx=1,
        called="train",
        stage="training",
        num_train_batches=5,
        num_epochs=3,
        num_val_batches=4,
    )
    maps.training.create_split(state.split_idx)

    logger = LoggerCallback()

    comp = ComputationalConfig(gpu=False)
    with caplog.at_level(logging.INFO):
        logger.on_train_start(maps=maps, state=state, computational=comp)
    assert "Beginning of training on split 1" in caplog.text
    assert "Computational configuration: gpu=False" in caplog.text

    state.current_epoch = 2
    stdout_capture = io.StringIO()
    with contextlib.redirect_stdout(stdout_capture), caplog.at_level(logging.INFO):
        logger.on_epoch_start(state=state)
    assert "Epoch 2/3: " in stdout_capture.getvalue()
    assert "Beginning of epoch 2" in caplog.text
    assert logger._train_progress_bar.total == 5
    assert logger._train_progress_bar.initial == 1
    assert logger._train_progress_bar.desc == "Epoch 2/3"
    assert logger._train_progress_bar.unit == "batch"

    caplog.clear()
    state.current_train_batch = 3
    with caplog.at_level(logging.DEBUG):
        logger.on_batch_start(state=state)
    assert "Batch 3 loaded" in caplog.text

    MODEL.get_loss_functions.return_value = {"abc": Mock()}
    logger.on_backward_step_start(state=state, loss=torch.tensor([1.1]), model=MODEL)
    assert logger._train_progress_bar.postfix == "abc=1.1"

    with caplog.at_level(logging.DEBUG):
        logger.on_batch_end(state=state)
    assert "Processing of batch 3 completed" in caplog.text
    assert logger._train_progress_bar.n == 3
    assert not [r for r in caplog.records if r.levelname == logging.INFO]

    state.stage == "evaluation"
    with contextlib.redirect_stdout(stdout_capture), caplog.at_level(logging.INFO):
        logger.on_validation_start(state=state, maps=maps)
    assert "Validation: " in stdout_capture.getvalue()
    assert "Beginning of validation" in caplog.text
    assert logger._val_progress_bar.total == 4
    assert logger._val_progress_bar.initial == 1
    assert logger._val_progress_bar.desc == "Validation"
    assert logger._val_progress_bar.unit == "batch"

    caplog.clear()
    with caplog.at_level(logging.INFO):
        logger.on_validation_end(state=state)
    assert "End of validation" in caplog.text
    assert "Validation metrics saved" not in caplog.text
    assert logger._val_progress_bar.disable

    state.stage == "training"
    with caplog.at_level(logging.INFO):
        logger.on_epoch_end(state=state)
    assert "Epoch 2 completed" in caplog.text
    assert logger._train_progress_bar.disable

    state.current_epoch = 3
    logger.on_epoch_start(state=state)
    assert logger._train_progress_bar.total == 5

    with caplog.at_level(logging.INFO):
        logger.on_train_end(state=state)
    assert "Training completed successfully (stopped after 3 epochs)" in caplog.text
    assert (
        f"All results, logs, and model checkpoints are saved in {tmp_path / 'training' / 'split-1'}"
        in caplog.text
    )

    # files
    run_name = maps.exec.runs_list[0]
    assert run_name.startswith("train_")
    run_dir = maps.exec.runs[run_name]
    f = maps.load_file(run_dir.debug)
    assert "Batch" in f
    assert "Epoch" not in f
    f = maps.load_file(run_dir.outputs)
    assert "Batch" not in f
    assert "Epoch" in f
    f = maps.load_file(run_dir.errors)
    assert len(f) == 0
    time.sleep(1)

    # disabled
    logger = LoggerCallback(progress_bar=False, debug=False)
    logger.on_train_start(maps=maps, state=state, computational=comp)
    logger.on_epoch_start(state=state)
    assert logger._train_progress_bar.disable
    logger.on_validation_start(state=state, maps=maps)
    assert logger._val_progress_bar.disable

    run_dir = maps.exec.runs[maps.exec.runs_list[1]]
    assert not run_dir.debug.exists()
    assert run_dir.outputs.exists()

    # don't save
    logger = LoggerCallback(save_logs=False)
    logger.on_train_start(maps=maps, state=state, computational=comp)
    assert len(maps.exec.runs_list) == 2


def test_validate(caplog, tmp_path):
    shutil.copytree(MAPS_PATH, tmp_path, dirs_exist_ok=True)
    maps = Maps(tmp_path)
    maps.exec.remove(non_empty_ok=True)
    state = TrainerState(
        split_idx=1,
        called="validate",
        stage="evaluation",
        num_val_batches=4,
    )
    maps.training.create_split(state.split_idx)

    logger = LoggerCallback()

    stdout_capture = io.StringIO()
    with contextlib.redirect_stdout(stdout_capture), caplog.at_level(logging.INFO):
        logger.on_validation_start(state=state, maps=maps)
    assert "Validation: " in stdout_capture.getvalue()
    assert "Beginning of validation" in caplog.text
    assert logger._val_progress_bar.total == 4
    assert logger._val_progress_bar.initial == 1
    assert logger._val_progress_bar.desc == "Validation"
    assert logger._val_progress_bar.unit == "batch"

    caplog.clear()
    state.current_val_batch = 2
    with caplog.at_level(logging.DEBUG):
        logger.on_batch_start(state=state)
    assert "Batch 2 loaded" in caplog.text

    with caplog.at_level(logging.DEBUG):
        logger.on_batch_end(state=state)
    assert "Processing of batch 2 completed" in caplog.text
    assert logger._val_progress_bar.n == 2
    assert not [r for r in caplog.records if r.levelname == logging.INFO]

    caplog.clear()
    with caplog.at_level(logging.INFO):
        logger.on_validation_end(state=state)
    assert "End of validation" in caplog.text
    assert (
        f"Validation metrics saved in {tmp_path / 'training' / 'split-1' / 'models'}"
        in caplog.text
    )
    assert logger._val_progress_bar.disable

    # files
    run_name = maps.exec.runs_list[0]
    assert run_name.startswith("validate_")
    run_dir = maps.exec.runs[run_name]
    f = maps.load_file(run_dir.debug)
    assert "Batch" in f
    assert "validation" not in f
    f = maps.load_file(run_dir.outputs)
    assert "Batch" not in f
    assert "validation" in f
    f = maps.load_file(run_dir.errors)
    assert len(f) == 0
    time.sleep(1)

    # with checkpoint
    state.split_idx = 0
    maps.training.splits[1].remove(non_empty_ok=True)
    maps.read()
    logger = LoggerCallback(progress_bar=False, debug=False)
    logger.on_validation_start(state=state, maps=maps, model_checkpoint="best-loss")
    assert logger._val_progress_bar.disable

    caplog.clear()
    with caplog.at_level(logging.INFO):
        logger.on_validation_end(state=state)
    assert (
        f"Validation metrics saved in {tmp_path / 'training' / 'split-0' / 'models' / 'best_models' / 'best-loss'}"
        in caplog.text
    )

    run_dir = maps.exec.runs[maps.exec.runs_list[1]]
    assert not run_dir.debug.exists()
    assert run_dir.outputs.exists()

    # don't save
    logger = LoggerCallback(save_logs=False)
    logger.on_validation_start(state=state, maps=maps)
    assert len(maps.exec.runs_list) == 2


def test_test(caplog, tmp_path):
    shutil.copytree(MAPS_PATH, tmp_path, dirs_exist_ok=True)
    maps = Maps(tmp_path)
    maps.exec.remove(non_empty_ok=True)
    maps.read()

    state = TrainerState(
        split_idx=0,
        called="test",
        stage="evaluation",
        num_test_batches=6,
    )

    logger = LoggerCallback()

    stdout_capture = io.StringIO()
    with contextlib.redirect_stdout(stdout_capture), caplog.at_level(logging.INFO):
        logger.on_test_start(
            state=state, maps=maps, model_checkpoint="split-0_best-loss", group_name="X"
        )
    assert "Test: " in stdout_capture.getvalue()
    assert "Beginning of test" in caplog.text
    assert logger._test_progress_bar.total == 6
    assert logger._test_progress_bar.initial == 1
    assert logger._test_progress_bar.desc == "Test"
    assert logger._test_progress_bar.unit == "batch"

    caplog.clear()
    state.current_test_batch = 2
    with caplog.at_level(logging.DEBUG):
        logger.on_batch_start(state=state)
    assert "Batch 2 loaded" in caplog.text

    with caplog.at_level(logging.DEBUG):
        logger.on_batch_end(state=state)
    assert "Processing of batch 2 completed" in caplog.text
    assert logger._test_progress_bar.n == 2
    assert not [r for r in caplog.records if r.levelname == logging.INFO]

    caplog.clear()
    with caplog.at_level(logging.INFO):
        logger.on_test_end(state=state)
    assert "End of test" in caplog.text
    assert (
        f"Test metrics saved in {tmp_path / 'test' / 'group-X' / 'results' / 'split-0' / 'best-loss'}"
        in caplog.text
    )
    assert logger._test_progress_bar.disable

    # files
    run_name = maps.exec.runs_list[0]
    assert run_name.startswith("test_")
    run_dir = maps.exec.runs[run_name]
    f = maps.load_file(run_dir.debug)
    assert "Batch" in f
    assert "test" not in f
    f = maps.load_file(run_dir.outputs)
    assert "Batch" not in f
    assert "test" in f
    f = maps.load_file(run_dir.errors)
    assert len(f) == 0
    time.sleep(1)

    # disable
    logger = LoggerCallback(progress_bar=False, debug=False)
    logger.on_test_start(
        state=state, maps=maps, model_checkpoint="split-0_best-loss", group_name="X"
    )
    assert logger._test_progress_bar.disable

    run_dir = maps.exec.runs[maps.exec.runs_list[1]]
    assert not run_dir.debug.exists()
    assert run_dir.outputs.exists()

    # don't save
    logger = LoggerCallback(save_logs=False)
    logger.on_test_start(
        state=state, maps=maps, model_checkpoint="split-0_best-loss", group_name="X"
    )
    assert len(maps.exec.runs_list) == 2


def test_predict(caplog, tmp_path):
    shutil.copytree(MAPS_PATH, tmp_path, dirs_exist_ok=True)
    maps = Maps(tmp_path)
    maps.exec.remove(non_empty_ok=True)
    maps.read()

    state = TrainerState(
        split_idx=0,
        called="predict",
        stage="prediction",
        num_pred_batches=5,
    )

    logger = LoggerCallback()

    stdout_capture = io.StringIO()
    with contextlib.redirect_stdout(stdout_capture), caplog.at_level(logging.INFO):
        logger.on_predict_start(
            state=state, maps=maps, model_checkpoint="split-0_best-loss", group_name="X"
        )
    assert "Prediction: " in stdout_capture.getvalue()
    assert "Beginning of prediction" in caplog.text
    assert logger._predict_progress_bar.total == 5
    assert logger._predict_progress_bar.initial == 1
    assert logger._predict_progress_bar.desc == "Prediction"
    assert logger._predict_progress_bar.unit == "batch"

    caplog.clear()
    state.current_pred_batch = 2
    with caplog.at_level(logging.DEBUG):
        logger.on_batch_start(state=state)
    assert "Batch 2 loaded" in caplog.text

    with caplog.at_level(logging.DEBUG):
        logger.on_batch_end(state=state)
    assert "Processing of batch 2 completed" in caplog.text
    assert logger._predict_progress_bar.n == 2
    assert not [r for r in caplog.records if r.levelname == logging.INFO]

    caplog.clear()
    with caplog.at_level(logging.INFO):
        logger.on_predict_end(state=state)
    assert "End of prediction" in caplog.text
    assert (
        f"Predictions saved in {tmp_path / 'prediction' / 'group-X' / 'results' / 'split-0' / 'best-loss'}"
        in caplog.text
    )
    assert logger._predict_progress_bar.disable

    # files
    run_name = maps.exec.runs_list[0]
    assert run_name.startswith("predict_")
    run_dir = maps.exec.runs[run_name]
    f = maps.load_file(run_dir.debug)
    assert "Batch" in f
    assert "Prediction" not in f
    f = maps.load_file(run_dir.outputs)
    assert "Batch" not in f
    assert "Prediction" in f
    f = maps.load_file(run_dir.errors)
    assert len(f) == 0
    time.sleep(1)

    # disable
    logger = LoggerCallback(progress_bar=False, debug=False)
    logger.on_predict_start(
        state=state, maps=maps, model_checkpoint="split-0_best-loss", group_name="X"
    )
    assert logger._predict_progress_bar.disable

    run_dir = maps.exec.runs[maps.exec.runs_list[1]]
    assert not run_dir.debug.exists()
    assert run_dir.outputs.exists()

    # don't save
    logger = LoggerCallback(save_logs=False)
    logger.on_predict_start(
        state=state, maps=maps, model_checkpoint="split-0_best-loss", group_name="X"
    )
    assert len(maps.exec.runs_list) == 2


def test_state_dict():
    logger = LoggerCallback()
    logger.load_state_dict(logger.state_dict())


def test_from_to_dict():
    logger = LoggerCallback(progress_bar=False, save_logs=False, debug=False)
    new_logger = LoggerCallback.from_dict(logger.to_dict())
    assert not new_logger.config.progress_bar
    assert not new_logger.config.save_logs
    assert not new_logger.config.debug
