import contextlib
import io
import logging
import shutil
import time
from pathlib import Path
from unittest.mock import MagicMock

import torch

from clinicadl.callbacks import LoggerCallback
from clinicadl.io import Maps
from clinicadl.train import ComputationalConfig, TrainerState

MAPS_PATH = Path(__file__).parents[2] / "resources" / "maps_example"

MODEL = MagicMock(name="model")
MODEL.get_summary.return_value = "abc"

SAMPLE = MagicMock()
SAMPLE.image.tensor.return_value = torch.randn(1, 2, 2, 2)
BATCH = [SAMPLE]

SPLIT = MagicMock()
SPLIT.train_dataset = MagicMock()
SPLIT.val_dataset = MagicMock()
SPLIT.train_dataset.__len__.return_value = 10
SPLIT.val_dataset.__len__.return_value = 5


def test_trainer(tmp_path):
    maps = Maps(tmp_path)
    maps.create(overwrite=True)
    maps.training.create_split(0)
    state = TrainerState(
        split_idx=0,
        called="train",
        stage="training",
    )
    SPLIT.index = state.split_idx

    logger = LoggerCallback()

    logger.on_trainer_init(model=MODEL, maps=maps)
    with open(maps.architecture_log, "r") as f:
        assert "model" in f.read()

    logger.on_train_start(
        maps=maps,
        split=SPLIT,
        state=state,
        computational=ComputationalConfig(gpu=False),
    )

    logger.on_batch_start(
        model=MODEL,
        maps=maps,
        state=state,
        batch=BATCH,
    )
    with open(maps.nn_summary_txt, "r") as f:
        assert "abc" == f.read()


def test_train(caplog, tmp_path):
    shutil.copytree(MAPS_PATH, tmp_path, dirs_exist_ok=True)
    maps = Maps(tmp_path)
    maps.exec.remove(non_empty_ok=True)
    state = TrainerState(
        split_idx=2,
        called="train",
        stage="training",
        num_train_batches=5,
        num_epochs=3,
        num_val_batches=4,
    )
    maps.training.create_split(state.split_idx)
    SPLIT.index = state.split_idx

    logger = LoggerCallback()
    logger.on_trainer_init(model=MODEL, maps=maps)

    comp = ComputationalConfig(gpu=False)
    with caplog.at_level(logging.INFO):
        logger.on_train_start(maps=maps, split=SPLIT, state=state, computational=comp)
    assert f"Beginning of training on split {state.split_idx}" in caplog.text
    assert "Computational configuration: gpu=False" in caplog.text
    assert maps.training.splits[state.split_idx].summary_log.is_file()
    with open(maps.training.splits[state.split_idx].summary_log, "r") as f:
        content = f.read()
    assert "Date:" in content
    assert "Trained with 10 samples" in content
    with open(maps.summary_log, "r") as f:
        assert f"- {state.split_idx}" in f.read()

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
        logger.on_batch_start(state=state, maps=maps, batch=BATCH, model=MODEL)
    assert "Batch 3 loaded" in caplog.text

    MODEL.get_loss_functions.return_value = {"abc": MagicMock()}
    logger.on_backward_step_start(state=state, loss=torch.tensor([1.1]), model=MODEL)
    assert logger._train_progress_bar.postfix == "abc=1.1"

    with caplog.at_level(logging.DEBUG):
        logger.on_batch_end(state=state)
    assert "Processing of batch 3 completed" in caplog.text
    assert logger._train_progress_bar.n == 3
    assert not [r for r in caplog.records if r.levelname == logging.INFO]

    state.stage = "evaluation"
    with contextlib.redirect_stdout(stdout_capture), caplog.at_level(logging.INFO):
        logger.on_validation_start(state=state)
    assert "Validation: " in stdout_capture.getvalue()
    assert "Beginning of validation" in caplog.text
    assert logger._val_progress_bar.total == 4
    assert logger._val_progress_bar.initial == 1
    assert logger._val_progress_bar.desc == "Validation"
    assert logger._val_progress_bar.unit == "batch"

    caplog.clear()
    with caplog.at_level(logging.INFO):
        logger.on_validation_end()
    assert "End of validation" in caplog.text
    assert "Validation metrics saved" not in caplog.text
    assert logger._val_progress_bar.disable

    state.stage = "training"
    with caplog.at_level(logging.INFO):
        logger.on_epoch_end(state=state)
    assert "Epoch 2 completed" in caplog.text
    assert logger._train_progress_bar.disable

    state.current_epoch = 3
    logger.on_epoch_start(state=state)
    assert logger._train_progress_bar.total == 5

    log = logging.getLogger("clinicadl.logger_test")
    log.warning("a warning")

    with caplog.at_level(logging.INFO):
        logger.on_train_end(state=state)
    assert "Training completed successfully (stopped after 3 epochs)" in caplog.text
    assert (
        f"All results, logs, and model checkpoints are saved in {tmp_path / 'training' / f'split-{state.split_idx}'}"
        in caplog.text
    )
    with open(maps.training.splits[state.split_idx].summary_log, "r") as f:
        assert "Training completed" in f.read()

    log.warning("a second warning")

    # files
    run_name = maps.exec.runs_list[0]
    assert run_name.startswith("train_")
    run_dir = maps.exec.runs[run_name]
    f = maps.open_file(run_dir.debug_log)
    assert "Batch" in f
    assert "Epoch" not in f
    f = maps.open_file(run_dir.info_log)
    assert "Batch" not in f
    assert "Epoch" in f
    assert "a warning" in f
    f = maps.open_file(run_dir.error_log)
    assert len(f) == 0
    f = maps.open_file(maps.training.splits[state.split_idx].warning_log)
    assert "Batch" not in f
    assert "Epoch" not in f
    assert "a warning" in f
    assert "a second warning" not in f
    time.sleep(1)

    # disabled
    logger = LoggerCallback(progress_bar=False, debug=False)
    logger.on_trainer_init(model=MODEL, maps=maps)
    logger.on_train_start(maps=maps, state=state, computational=comp, split=SPLIT)
    logger.on_epoch_start(state=state)
    assert logger._train_progress_bar.disable
    logger.on_validation_start(state=state)
    assert logger._val_progress_bar.disable

    run_dir = maps.exec.runs[maps.exec.runs_list[1]]
    assert not run_dir.debug_log.exists()
    assert run_dir.info_log.exists()
    time.sleep(1)

    # don't save
    maps.training.splits[state.split_idx].warning_log.unlink()
    logger = LoggerCallback(save_logs=False)
    logger.on_trainer_init(model=MODEL, maps=maps)
    logger.on_train_start(maps=maps, state=state, computational=comp, split=SPLIT)
    log.warning("a warning")
    assert len(maps.exec.runs_list) == 2
    assert not maps.training.splits[state.split_idx].warning_log.is_file()


def test_validate(caplog, tmp_path):
    shutil.copytree(MAPS_PATH, tmp_path, dirs_exist_ok=True)
    maps = Maps(tmp_path)
    maps.read()
    maps.exec.remove(non_empty_ok=True)

    state = TrainerState(
        split_idx=0,
        called="validate",
        stage="evaluation",
        num_val_batches=4,
    )
    SPLIT.index = state.split_idx

    logger = LoggerCallback()
    logger.on_trainer_init(model=MODEL, maps=maps)

    stdout_capture = io.StringIO()
    with contextlib.redirect_stdout(stdout_capture), caplog.at_level(logging.INFO):
        logger.on_validate_start(state=state, maps=maps, model_checkpoint="best-loss")
    assert "Validation: " in stdout_capture.getvalue()
    assert (
        f"Beginning of validation of checkpoint 'best-loss' on split {SPLIT.index}"
        in caplog.text
    )
    assert logger._val_progress_bar.total == 4
    assert logger._val_progress_bar.initial == 1
    assert logger._val_progress_bar.desc == "Validation"
    assert logger._val_progress_bar.unit == "batch"

    caplog.clear()
    state.current_val_batch = 2
    with caplog.at_level(logging.DEBUG):
        logger.on_batch_start(state=state, maps=maps, batch=BATCH, model=MODEL)
    assert "Batch 2 loaded" in caplog.text

    with caplog.at_level(logging.DEBUG):
        logger.on_batch_end(state=state)
    assert "Processing of batch 2 completed" in caplog.text
    assert logger._val_progress_bar.n == 2
    assert not [r for r in caplog.records if r.levelname == logging.INFO]

    log = logging.getLogger("clinicadl.logger_test")
    log.warning("a warning")

    caplog.clear()
    with caplog.at_level(logging.INFO):
        logger.on_validate_end()
    assert "End of validation" in caplog.text
    assert (
        f"Validation metrics saved in {tmp_path / 'training' / f'split-{state.split_idx}' / 'models' / 'best_models' / 'best-loss'}"
        in caplog.text
    )
    assert logger._val_progress_bar.disable

    log.warning("a second warning")

    # files
    run_name = maps.exec.runs_list[0]
    assert run_name.startswith("validate_")
    run_dir = maps.exec.runs[run_name]
    f = maps.open_file(run_dir.debug_log)
    assert "Batch" in f
    assert "validation" not in f
    f = maps.open_file(run_dir.info_log)
    assert "Batch" not in f
    assert "validation" in f
    assert "a warning" in f
    f = maps.open_file(run_dir.error_log)
    assert len(f) == 0
    f = maps.open_file(
        maps.training.splits[state.split_idx]
        .models.best_models.metrics["loss"]
        .warning_log
    )
    assert "Batch" not in f
    assert "validation" not in f
    assert "a warning" in f
    assert "a second warning" not in f
    time.sleep(1)

    # disable
    state.split_idx = 0
    logger = LoggerCallback(progress_bar=False, debug=False)
    logger.on_trainer_init(model=MODEL, maps=maps)
    logger.on_validate_start(state=state, maps=maps, model_checkpoint="best-loss")
    assert logger._val_progress_bar.disable

    log.warning("a third warning")

    run_dir = maps.exec.runs[maps.exec.runs_list[1]]
    assert not run_dir.debug_log.exists()
    assert run_dir.info_log.exists()
    time.sleep(1)

    # don't save
    maps.training.splits[state.split_idx].warning_log.unlink()
    logger = LoggerCallback(save_logs=False)
    logger.on_trainer_init(model=MODEL, maps=maps)
    logger.on_validate_start(state=state, maps=maps, model_checkpoint="best-loss")
    log.warning("a warning")
    assert len(maps.exec.runs_list) == 2
    assert not maps.training.splits[state.split_idx].warning_log.is_file()
    logger.on_validate_end()


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
    logger.on_trainer_init(model=MODEL, maps=maps)

    stdout_capture = io.StringIO()
    with contextlib.redirect_stdout(stdout_capture), caplog.at_level(logging.INFO):
        logger.on_test_start(
            state=state, maps=maps, model_checkpoint="split-0_best-loss", group_name="X"
        )
    assert "Test: " in stdout_capture.getvalue()
    assert "Beginning of test" in caplog.text
    with open(maps.summary_log, "r") as f:
        assert f"- X" in f.read()
    assert logger._test_progress_bar.total == 6
    assert logger._test_progress_bar.initial == 1
    assert logger._test_progress_bar.desc == "Test"
    assert logger._test_progress_bar.unit == "batch"

    caplog.clear()
    state.current_test_batch = 2
    with caplog.at_level(logging.DEBUG):
        logger.on_batch_start(state=state, maps=maps, batch=BATCH, model=MODEL)
    assert "Batch 2 loaded" in caplog.text

    with caplog.at_level(logging.DEBUG):
        logger.on_batch_end(state=state)
    assert "Processing of batch 2 completed" in caplog.text
    assert logger._test_progress_bar.n == 2
    assert not [r for r in caplog.records if r.levelname == logging.INFO]

    log = logging.getLogger("clinicadl.logger_test")
    log.warning("a warning")

    caplog.clear()
    with caplog.at_level(logging.INFO):
        logger.on_test_end(state=state)
    assert "End of test" in caplog.text
    assert (
        f"Test metrics saved in {tmp_path / 'test' / 'group-X' / 'results' / 'split-0' / 'best-loss'}"
        in caplog.text
    )
    assert logger._test_progress_bar.disable

    log.warning("a second warning")

    # files
    run_name = maps.exec.runs_list[0]
    assert run_name.startswith("test_")
    run_dir = maps.exec.runs[run_name]
    f = maps.open_file(run_dir.debug_log)
    assert "Batch" in f
    assert "test" not in f
    f = maps.open_file(run_dir.info_log)
    assert "Batch" not in f
    assert "test" in f
    assert "a warning" in f
    f = maps.open_file(run_dir.error_log)
    assert len(f) == 0
    f = maps.open_file(
        maps.test.groups["X"].results.splits[0].models["best-loss"].warning_log
    )
    assert "Batch" not in f
    assert "test" not in f
    assert "a warning" in f
    assert "a second warning" not in f
    time.sleep(1)

    # disable
    logger = LoggerCallback(progress_bar=False, debug=False)
    logger.on_trainer_init(model=MODEL, maps=maps)
    logger.on_test_start(
        state=state, maps=maps, model_checkpoint="split-0_best-loss", group_name="X"
    )
    assert logger._test_progress_bar.disable
    logger.on_test_end(state=state)

    run_dir = maps.exec.runs[maps.exec.runs_list[1]]
    assert not run_dir.debug_log.exists()
    assert run_dir.info_log.exists()
    time.sleep(1)

    # don't save
    maps.test.groups["X"].results.splits[0].models["best-loss"].warning_log.unlink()
    logger = LoggerCallback(save_logs=False)
    logger.on_trainer_init(model=MODEL, maps=maps)
    logger.on_test_start(
        state=state, maps=maps, model_checkpoint="split-0_best-loss", group_name="X"
    )
    log.warning("a warning")
    logger.on_test_end(state=state)
    assert (
        not maps.test.groups["X"]
        .results.splits[0]
        .models["best-loss"]
        .warning_log.is_file()
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
    logger.on_trainer_init(model=MODEL, maps=maps)

    stdout_capture = io.StringIO()
    with contextlib.redirect_stdout(stdout_capture), caplog.at_level(logging.INFO):
        logger.on_predict_start(
            state=state, maps=maps, model_checkpoint="split-0_best-loss", group_name="X"
        )
    assert "Prediction: " in stdout_capture.getvalue()
    assert "Beginning of prediction" in caplog.text
    with open(maps.summary_log, "r") as f:
        assert f"- X" in f.read()
    assert logger._predict_progress_bar.total == 5
    assert logger._predict_progress_bar.initial == 1
    assert logger._predict_progress_bar.desc == "Prediction"
    assert logger._predict_progress_bar.unit == "batch"

    caplog.clear()
    state.current_pred_batch = 2
    with caplog.at_level(logging.DEBUG):
        logger.on_batch_start(state=state, maps=maps, batch=BATCH, model=MODEL)
    assert "Batch 2 loaded" in caplog.text

    with caplog.at_level(logging.DEBUG):
        logger.on_batch_end(state=state)
    assert "Processing of batch 2 completed" in caplog.text
    assert logger._predict_progress_bar.n == 2
    assert not [r for r in caplog.records if r.levelname == logging.INFO]

    log = logging.getLogger("clinicadl.logger_test")
    log.warning("a warning")

    caplog.clear()
    with caplog.at_level(logging.INFO):
        logger.on_predict_end(state=state)
    assert "End of prediction" in caplog.text
    assert (
        f"Predictions saved in {tmp_path / 'prediction' / 'group-X' / 'results' / 'split-0' / 'best-loss'}"
        in caplog.text
    )
    assert logger._predict_progress_bar.disable

    log.warning("a second warning")

    # files
    run_name = maps.exec.runs_list[0]
    assert run_name.startswith("predict_")
    run_dir = maps.exec.runs[run_name]
    f = maps.open_file(run_dir.debug_log)
    assert "Batch" in f
    assert "Prediction" not in f
    f = maps.open_file(run_dir.info_log)
    assert "Batch" not in f
    assert "Prediction" in f
    f = maps.open_file(run_dir.error_log)
    assert len(f) == 0
    f = maps.open_file(
        maps.prediction.groups["X"].results.splits[0].models["best-loss"].warning_log
    )
    assert "Batch" not in f
    assert "Prediction" not in f
    assert "a warning" in f
    assert "a second warning" not in f
    time.sleep(1)

    # disable
    logger = LoggerCallback(progress_bar=False, debug=False)
    logger.on_trainer_init(model=MODEL, maps=maps)
    logger.on_predict_start(
        state=state, maps=maps, model_checkpoint="split-0_best-loss", group_name="X"
    )
    assert logger._predict_progress_bar.disable
    logger.on_predict_end(state=state)

    run_dir = maps.exec.runs[maps.exec.runs_list[1]]
    assert not run_dir.debug_log.exists()
    assert run_dir.info_log.exists()
    time.sleep(1)

    # don't save
    maps.prediction.groups["X"].results.splits[0].models[
        "best-loss"
    ].warning_log.unlink()
    logger = LoggerCallback(save_logs=False)
    logger.on_trainer_init(model=MODEL, maps=maps)
    logger.on_predict_start(
        state=state, maps=maps, model_checkpoint="split-0_best-loss", group_name="X"
    )
    log.warning("a warning")
    logger.on_predict_end(state=state)
    assert (
        not maps.prediction.groups["X"]
        .results.splits[0]
        .models["best-loss"]
        .warning_log.is_file()
    )
    assert len(maps.exec.runs_list) == 2


def test_on_exception(caplog, tmp_path):
    shutil.copytree(MAPS_PATH, tmp_path, dirs_exist_ok=True)
    maps = Maps(tmp_path)
    maps.exec.remove(non_empty_ok=True)
    state = TrainerState(
        split_idx=2,
        called="train",
        stage="training",
    )
    maps.training.create_split(state.split_idx)
    SPLIT.index = state.split_idx
    comp = ComputationalConfig(gpu=False)

    logger = LoggerCallback()

    logger.on_trainer_init(model=MODEL, maps=maps)
    with caplog.at_level(logging.INFO):
        logger.on_exception(state=state)
    assert len(caplog.records) == 0

    # train
    logger.on_train_start(maps=maps, split=SPLIT, state=state, computational=comp)

    log = logging.getLogger("clinicadl.logger_test")
    log.warning("a warning")

    state.stage = "interrupted"
    log.exception(ValueError("an error"))
    with caplog.at_level(logging.INFO):
        logger.on_exception(state=state)
    log.warning("a second warning")

    with open(maps.training.splits[state.split_idx].summary_log, "r") as f:
        assert "Training interrupted" in f.read()

    run_name = maps.exec.runs_list[0]
    run_dir = maps.exec.runs[run_name]
    f = maps.open_file(run_dir.error_log)
    assert "an error" in f
    f = maps.open_file(run_dir.info_log)
    assert "a warning" in f
    assert "a second warning" not in f

    assert (
        f"An exception occurred. To debug, check the logs in {run_dir}" in caplog.text
    )

    # don't save files
    logger = LoggerCallback(save_logs=False)
    logger.on_trainer_init(model=MODEL, maps=maps)
    logger.on_train_start(maps=maps, split=SPLIT, state=state, computational=comp)

    caplog.clear()
    with caplog.at_level(logging.INFO):
        logger.on_exception(state=state)

    assert "An exception occurred. To debug, check the logs in" not in caplog.text


def test_resume(caplog, tmp_path):
    shutil.copytree(MAPS_PATH, tmp_path, dirs_exist_ok=True)
    maps = Maps(tmp_path)
    maps.read()
    state = TrainerState(
        split_idx=0,
        num_epochs=4,
    )
    SPLIT.index = state.split_idx

    logger = LoggerCallback(save_logs=False)
    logger.on_trainer_init(model=MODEL, maps=maps)

    comp = ComputationalConfig(gpu=False)
    with caplog.at_level(logging.INFO):
        logger.on_resume(maps=maps, split=SPLIT, state=state, computational=comp)
    assert f"Resuming training on split {state.split_idx} from epoch 4" in caplog.text
    assert "Computational configuration: gpu=False" in caplog.text

    state.current_epoch = 4
    caplog.clear()
    with caplog.at_level(logging.INFO):
        logger.on_train_end(state=state)
    assert "Training completed successfully (stopped after 4 epochs)" in caplog.text
    assert (
        f"All results, logs, and model checkpoints are saved in {tmp_path / 'training' / f'split-{state.split_idx}'}"
        in caplog.text
    )


def test_from_to_dict():
    logger = LoggerCallback(progress_bar=False, save_logs=False, debug=False)
    new_logger = LoggerCallback.from_dict(logger.to_dict())
    assert not new_logger.config.progress_bar
    assert not new_logger.config.save_logs
    assert not new_logger.config.debug
