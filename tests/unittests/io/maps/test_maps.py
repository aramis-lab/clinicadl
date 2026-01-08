import os
import re
import shutil
from pathlib import Path

import pandas as pd
import pytest
import torch

from clinicadl.io.maps import Maps

REFERENCE_MAPS = Path(__file__).parents[2] / "resources" / "maps_example"
MINIMAL_MAPS = Path(__file__).parents[2] / "resources" / "maps_minimal"


def _build_pattern(prefix: Path, suffix: str) -> re.Pattern:
    sep = re.escape(os.sep)
    prefix = re.escape(str(prefix))
    suffix = re.escape(suffix)

    pattern = (
        rf"^{prefix}{sep}"
        r"run\-train_\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}"
        rf"{sep}{suffix}\.log$"
    )
    print(pattern)
    return re.compile(pattern)


def test_maps(tmp_path: Path):
    maps_path = tmp_path / "maps"
    maps = Maps(maps_path)

    maps.create()
    assert (maps_path / "summary.log").is_file()
    assert (maps_path / "environment.txt").is_file()

    assert maps.architecture_log == (maps_path / "architecture.log")
    assert maps.model_json == (maps_path / "model.json")
    assert maps.metrics_json == (maps_path / "metrics.json")
    assert maps.nn_summary_txt == (maps_path / "nn_summary.txt")
    assert maps.callbacks_json == (maps_path / "callbacks.json")

    # runs
    run = maps.exec.create_run(process_called="train")
    pattern = r"^train_\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}$"
    assert re.match(pattern, run)
    assert re.match(
        _build_pattern(prefix=maps_path / "exec", suffix="error"),
        str(maps.exec.runs[run].error_log),
    )
    assert re.match(
        _build_pattern(prefix=maps_path / "exec", suffix="debug"),
        str(maps.exec.runs[run].debug_log),
    )
    assert re.match(
        _build_pattern(prefix=maps_path / "exec", suffix="info"),
        str(maps.exec.runs[run].info_log),
    )

    # training
    maps.training.create(exist_ok=True)

    assert maps.training.optimization_json == (
        maps_path / "training" / "optimization.json"
    )

    # training - splits
    maps.training.create_split(0)
    assert maps.training.splits_list == [0]
    assert (maps_path / "training" / "split-0").exists()

    assert maps.training.splits[0].computational_json == (
        maps_path / "training" / "split-0" / "computational.json"
    )
    assert maps.training.splits[0].summary_log == (
        maps_path / "training" / "split-0" / "summary.log"
    )
    assert maps.training.splits[0].warning_log == (
        maps_path / "training" / "split-0" / "warning.log"
    )

    # training - splits - validation_metrics
    assert maps.training.splits[0].validation_metrics.aggregated == (
        maps_path / "training" / "split-0" / "validation_metrics" / "aggregated.tsv"
    )
    assert maps.training.splits[0].validation_metrics.details == (
        maps_path / "training" / "split-0" / "validation_metrics" / "details.tsv"
    )

    # training - splits - logs
    assert maps.training.splits[0].logs.training_loss == (
        maps_path / "training" / "split-0" / "logs" / "training_loss.tsv"
    )
    assert maps.training.splits[0].logs.learning_rates == (
        maps_path / "training" / "split-0" / "logs" / "learning_rates"
    )

    # training - splits - tmp
    maps.training.splits[0].tmp.create_epoch(0)
    assert maps.training.splits[0].tmp.epochs_list == [0]
    assert (maps_path / "training" / "split-0" / "tmp" / "epoch-0").exists()

    assert maps.training.splits[0].tmp.epochs[0].callbacks == (
        maps_path / "training" / "split-0" / "tmp" / "epoch-0" / "callbacks"
    )
    assert maps.training.splits[0].tmp.epochs[0].state == (
        maps_path / "training" / "split-0" / "tmp" / "epoch-0" / "state.json"
    )
    assert maps.training.splits[0].tmp.epochs[0].scaler == (
        maps_path / "training" / "split-0" / "tmp" / "epoch-0" / "scaler.json"
    )
    assert maps.training.splits[0].tmp.epochs[0].model == (
        maps_path / "training" / "split-0" / "tmp" / "epoch-0" / "model.pth.tar"
    )

    # training - splits - tmp - validation_metrics
    assert maps.training.splits[0].tmp.epochs[0].validation_metrics.details == (
        maps_path
        / "training"
        / "split-0"
        / "tmp"
        / "epoch-0"
        / "validation_metrics"
        / "details.tsv"
    )
    assert maps.training.splits[0].tmp.epochs[0].validation_metrics.aggregated == (
        maps_path
        / "training"
        / "split-0"
        / "tmp"
        / "epoch-0"
        / "validation_metrics"
        / "aggregated.tsv"
    )

    # training - splits - models

    # training - splits - models - best_models
    maps.training.splits[0].models.best_models.create_metric("loss")
    assert maps.training.splits[0].models.best_models.metrics_list == ["loss"]
    assert (
        maps_path / "training" / "split-0" / "models" / "best_models" / "best-loss"
    ).exists()
    assert maps.training.splits[0].models.best_models.metrics["loss"].model == (
        maps_path
        / "training"
        / "split-0"
        / "models"
        / "best_models"
        / "best-loss"
        / "model.pth.tar"
    )
    assert maps.training.splits[0].models.best_models.metrics["loss"].warning_log == (
        maps_path
        / "training"
        / "split-0"
        / "models"
        / "best_models"
        / "best-loss"
        / "warning.log"
    )

    # training - splits - models - best_models - validation_metrics
    assert maps.training.splits[0].models.best_models.metrics[
        "loss"
    ].validation_metrics.aggregated == (
        maps_path
        / "training"
        / "split-0"
        / "models"
        / "best_models"
        / "best-loss"
        / "validation_metrics"
        / "aggregated.tsv"
    )
    assert maps.training.splits[0].models.best_models.metrics[
        "loss"
    ].validation_metrics.details == (
        maps_path
        / "training"
        / "split-0"
        / "models"
        / "best_models"
        / "best-loss"
        / "validation_metrics"
        / "details.tsv"
    )

    # training - splits - models - checkpoints
    maps.training.splits[0].models.checkpoints.create_epoch(0)
    assert maps.training.splits[0].models.checkpoints.epochs_list == [0]
    assert (
        maps_path / "training" / "split-0" / "models" / "checkpoints" / "epoch-0"
    ).exists()

    assert maps.training.splits[0].models.checkpoints.epochs[0].model == (
        maps_path
        / "training"
        / "split-0"
        / "models"
        / "checkpoints"
        / "epoch-0"
        / "model.pth.tar"
    )
    assert maps.training.splits[0].models.checkpoints.epochs[0].warning_log == (
        maps_path
        / "training"
        / "split-0"
        / "models"
        / "checkpoints"
        / "epoch-0"
        / "warning.log"
    )

    # training - splits - models - checkpoints - validation_metrics
    assert maps.training.splits[0].models.checkpoints.epochs[
        0
    ].validation_metrics.aggregated == (
        maps_path
        / "training"
        / "split-0"
        / "models"
        / "checkpoints"
        / "epoch-0"
        / "validation_metrics"
        / "aggregated.tsv"
    )
    assert maps.training.splits[0].models.checkpoints.epochs[
        0
    ].validation_metrics.details == (
        maps_path
        / "training"
        / "split-0"
        / "models"
        / "checkpoints"
        / "epoch-0"
        / "validation_metrics"
        / "details.tsv"
    )

    # training - splits - models - final
    assert maps.training.splits[0].models.final.model == (
        maps_path / "training" / "split-0" / "models" / "final" / "model.pth.tar"
    )
    assert maps.training.splits[0].models.final.warning_log == (
        maps_path / "training" / "split-0" / "models" / "final" / "warning.log"
    )

    # training - splits - models - final - validation_metrics
    assert maps.training.splits[0].models.final.validation_metrics.aggregated == (
        maps_path
        / "training"
        / "split-0"
        / "models"
        / "final"
        / "validation_metrics"
        / "aggregated.tsv"
    )
    assert maps.training.splits[0].models.final.validation_metrics.details == (
        maps_path
        / "training"
        / "split-0"
        / "models"
        / "final"
        / "validation_metrics"
        / "details.tsv"
    )

    # training - data
    assert maps.training.data.data_tsv == (maps_path / "training" / "data" / "data.tsv")

    # training - data - train
    assert maps.training.data.train.dataloader_json == (
        maps_path / "training" / "data" / "train" / "dataloader.json"
    )
    assert maps.training.data.train.dataset_json == (
        maps_path / "training" / "data" / "train" / "dataset.json"
    )

    maps.training.data.train.create_split(0)
    assert maps.training.data.train.splits_list == [0]
    assert (maps_path / "training" / "data" / "train" / "split-0").exists()
    assert maps.training.data.train.splits[0].data_tsv == (
        maps_path / "training" / "data" / "train" / "split-0" / "data.tsv"
    )

    # training - data - validation
    assert maps.training.data.validation.dataset_json == (
        maps_path / "training" / "data" / "validation" / "dataset.json"
    )

    maps.training.data.validation.create_split(0)
    assert maps.training.data.validation.splits_list == [0]
    assert (maps_path / "training" / "data" / "validation" / "split-0").exists()
    assert maps.training.data.validation.splits[0].data_tsv == (
        maps_path / "training" / "data" / "validation" / "split-0" / "data.tsv"
    )

    # test
    maps.test.create_group("X")
    assert maps.test.groups_list == ["X"]
    assert (maps_path / "test" / "group-X").exists()
    assert maps.test.groups["X"]
    assert maps.test.groups["X"].data_tsv == (
        maps_path / "test" / "group-X" / "data.tsv"
    )
    assert maps.test.groups["X"].dataset_json == (
        maps_path / "test" / "group-X" / "dataset.json"
    )

    # test - group - splits
    maps.test.groups["X"].results.create_split(0)
    assert maps.test.groups["X"].results.splits_list == [0]
    assert (maps_path / "test" / "group-X" / "results" / "split-0").exists()

    # test - group - splits - model
    maps.test.groups["X"].results.splits[0].create_model("best-loss")
    assert maps.test.groups["X"].results.splits[0].models_list == ["best-loss"]
    assert (
        maps_path / "test" / "group-X" / "results" / "split-0" / "best-loss"
    ).exists()
    assert maps.test.groups["X"].results.splits[0].models["best-loss"].warning_log == (
        maps_path
        / "test"
        / "group-X"
        / "results"
        / "split-0"
        / "best-loss"
        / "warning.log"
    )

    # test - group - splits - model - metrics
    assert maps.test.groups["X"].results.splits[0].models[
        "best-loss"
    ].metrics.aggregated == (
        maps_path
        / "test"
        / "group-X"
        / "results"
        / "split-0"
        / "best-loss"
        / "metrics"
        / "aggregated.tsv"
    )
    assert maps.test.groups["X"].results.splits[0].models[
        "best-loss"
    ].metrics.details == (
        maps_path
        / "test"
        / "group-X"
        / "results"
        / "split-0"
        / "best-loss"
        / "metrics"
        / "details.tsv"
    )

    # prediction
    maps.prediction.create_group("X")
    assert maps.prediction.groups_list == ["X"]
    assert (maps_path / "prediction" / "group-X").exists()
    assert maps.prediction.groups["X"]
    assert maps.prediction.groups["X"].data_tsv == (
        maps_path / "prediction" / "group-X" / "data.tsv"
    )
    assert maps.prediction.groups["X"].dataset_json == (
        maps_path / "prediction" / "group-X" / "dataset.json"
    )

    # prediction - group - results - splits
    maps.prediction.groups["X"].results.create_split(0)
    assert maps.prediction.groups["X"].results.splits_list == [0]
    assert (maps_path / "prediction" / "group-X" / "results" / "split-0").exists()

    # prediction - group - splits - model
    maps.prediction.groups["X"].results.splits[0].create_model("best-loss")
    assert maps.prediction.groups["X"].results.splits[0].models_list == ["best-loss"]
    assert (
        maps_path / "prediction" / "group-X" / "results" / "split-0" / "best-loss"
    ).exists()
    assert maps.prediction.groups["X"].results.splits[0].models[
        "best-loss"
    ].warning_log == (
        maps_path
        / "prediction"
        / "group-X"
        / "results"
        / "split-0"
        / "best-loss"
        / "warning.log"
    )

    # prediction - group - splits - model - output_tsv
    assert maps.prediction.groups["X"].results.splits[0].models[
        "best-loss"
    ].output_tsv == (
        maps_path
        / "prediction"
        / "group-X"
        / "results"
        / "split-0"
        / "best-loss"
        / "output.tsv"
    )

    # prediction - group - splits - model - caps_output
    assert maps.prediction.groups["X"].results.splits[0].models[
        "best-loss"
    ].caps_output == (
        maps_path
        / "prediction"
        / "group-X"
        / "results"
        / "split-0"
        / "best-loss"
        / "caps_output"
    )

    with pytest.raises(
        FileExistsError,
        match=".* is not empty! To confirm that you want to delete it, pass non_empty_ok=True",
    ):
        maps.remove()
    maps.remove(non_empty_ok=True)

    assert not maps_path.exists()


def test_create(tmp_path: Path):
    maps_path = tmp_path / "maps"
    maps = Maps(maps_path)

    maps.create()
    maps.training.data.create()
    with open(maps.training.data.data_tsv, "w", encoding="utf-8") as f:
        print("", file=f)

    with pytest.raises(
        FileExistsError,
        match="Directory .* already exists. If it's ok, pass exist_ok=True. To overwrite it, pass overwrite=True.",
    ):
        maps.create()

    maps.create(exist_ok=True)
    assert maps.training.data.data_tsv.is_file()
    maps.create(overwrite=True)
    assert not maps.architecture_log.is_file()


def test_read(tmp_path):
    maps_path = tmp_path / "maps"
    shutil.copytree(REFERENCE_MAPS, maps_path)

    maps = Maps(maps_path)
    maps.read()
    assert (
        maps.training.splits[0]
        .models.best_models.metrics["loss"]
        .validation_metrics.aggregated.is_file()
    )
    assert (
        maps.training.splits[0]
        .models.checkpoints.epochs[0]
        .validation_metrics.aggregated.is_file()
    )
    assert maps.training.splits[0].tmp.epochs[0].validation_metrics.aggregated.is_file()
    assert maps.training.data.train.splits[0].data_tsv.is_file()
    assert maps.training.data.validation.splits[0].data_tsv.is_file()
    assert (
        maps.prediction.groups["X"]
        .results.splits[0]
        .models["best-loss"]
        .output_tsv.is_file()
    )
    assert (
        maps.test.groups["X"]
        .results.splits[0]
        .models["best-loss"]
        .metrics.aggregated.is_file()
    )
    assert maps.exec.runs["train_2025_12_31_23_59_59"].debug_log.is_file()

    maps_path = tmp_path / "minimal_maps"
    shutil.copytree(MINIMAL_MAPS, maps_path)
    maps = Maps(maps_path)
    maps.read()

    # mandatory files
    for root, _, files in os.walk(maps_path):
        for f in files:
            shutil.copytree(MINIMAL_MAPS, maps_path, dirs_exist_ok=True)
            os.remove(Path(root) / f)
            maps = Maps(maps_path)
            if f != ".gitignore":
                with pytest.raises(FileNotFoundError):
                    maps.read()

    # mandatory dirs
    shutil.copytree(MINIMAL_MAPS, maps_path, dirs_exist_ok=True)
    maps = Maps(maps_path)
    shutil.rmtree(maps.test.path)
    shutil.rmtree(maps.prediction.path)
    maps.read()

    shutil.rmtree(maps.training.data.train.splits[0].path)
    with pytest.raises(
        FileNotFoundError, match="split-0 not found in the training data .*"
    ):
        maps.read()

    shutil.copytree(MINIMAL_MAPS, maps_path, dirs_exist_ok=True)
    shutil.rmtree(maps.training.data.validation.splits[0].path)
    with pytest.raises(
        FileNotFoundError, match="split-0 not found in the validation data .*"
    ):
        maps.read()
    shutil.rmtree(maps.training.splits[0].path)
    maps.read()

    shutil.copytree(MINIMAL_MAPS, maps_path, dirs_exist_ok=True)
    shutil.rmtree(maps.training.data.train.path)
    with pytest.raises(FileNotFoundError):
        maps.read()

    shutil.copytree(MINIMAL_MAPS, maps_path, dirs_exist_ok=True)
    shutil.rmtree(maps.training.data.validation.path)
    with pytest.raises(FileNotFoundError):
        maps.read()

    shutil.copytree(MINIMAL_MAPS, maps_path, dirs_exist_ok=True)
    shutil.rmtree(maps.training.path)
    with pytest.raises(FileNotFoundError):
        maps.read()

    shutil.copytree(MINIMAL_MAPS, maps_path, dirs_exist_ok=True)
    shutil.rmtree(maps.training.splits[0].tmp.path)
    maps.read()


def test_load_file(tmp_path):
    maps_path = tmp_path / "maps"
    shutil.copytree(REFERENCE_MAPS, maps_path)

    maps = Maps(maps_path)
    maps.read()

    assert maps.load_file(maps.nn_summary_txt) == "test"
    assert maps.load_file(maps.summary_log) == "test"
    assert maps.load_file(maps.metrics_json) == {"test": True}
    pd.testing.assert_frame_equal(
        maps.load_file(maps.training.data.data_tsv), pd.DataFrame({"A": [0], "B": [0]})
    )
    torch.testing.assert_close(
        maps.load_file(maps.training.splits[0].models.checkpoints.epochs[0].model),
        torch.Tensor([0]),
    )

    with pytest.raises(FileNotFoundError, match=".* is not a file!"):
        maps.load_file(maps.training.splits[0].tmp.epochs[0].callbacks)


def test_save_file(tmp_path):
    maps_path = tmp_path / "maps"
    shutil.copytree(REFERENCE_MAPS, maps_path)

    maps = Maps(maps_path)
    maps.read()

    with pytest.raises(
        FileExistsError, match=".* exists! To overwrite it, pass overwrite=True."
    ):
        maps.save_file("abc", maps.nn_summary_txt)

    maps.save_file("abc", maps.nn_summary_txt, overwrite=True)
    assert maps.load_file(maps.nn_summary_txt) == "abc"

    maps.save_file("abc", maps.summary_log, overwrite=True)
    assert maps.load_file(maps.summary_log) == "abc"

    maps.save_file({"abc": True}, maps.metrics_json, overwrite=True)
    assert maps.load_file(maps.metrics_json) == {"abc": True}

    maps.save_file(
        pd.DataFrame({"X": [0], "Z": [0]}), maps.training.data.data_tsv, overwrite=True
    )
    pd.testing.assert_frame_equal(
        maps.load_file(maps.training.data.data_tsv), pd.DataFrame({"X": [0], "Z": [0]})
    )

    maps.save_file(
        torch.Tensor(0),
        maps.training.splits[0].models.checkpoints.epochs[0].model,
        overwrite=True,
    )

    with pytest.raises(IsADirectoryError, match=".* is not a valid file name!"):
        maps.save_file("abc", maps.training.splits[0].tmp.epochs[0].callbacks)
    with pytest.raises(
        ValueError,
        match=re.escape(
            "'.abc' files are not supported. "
            "The supported files in a MAPS directory are ['.json', '.log', '.txt', '.tsv', '.pth.tar']"
        ),
    ):
        maps.save_file("abc", "abc.abc")


def test_clear_tmp(tmp_path):
    maps_path = tmp_path / "maps"
    shutil.copytree(REFERENCE_MAPS, maps_path)

    maps = Maps(maps_path)
    maps.read()

    maps.training.splits[0].tmp.create_epoch(1)
    maps.training.splits[0].tmp.create_epoch(2)
    maps.training.splits[0].tmp.clear()
    assert not (maps_path / "training" / "split-0" / "tmp" / "epoch-1").exists()
    assert not (maps_path / "training" / "split-0" / "tmp" / "epoch-2").exists()
    maps.training.splits[0].tmp.create_epoch(1)
    maps.training.splits[0].tmp.create_epoch(2)
    maps.training.splits[0].tmp.clear(except_epoch=2)
    assert not (maps_path / "training" / "split-0" / "tmp" / "epoch-1").exists()
    assert (maps_path / "training" / "split-0" / "tmp" / "epoch-2").exists()
    assert maps.training.splits[0].tmp.epochs_list == [2]


def test_iterdir(tmp_path):
    maps_path = tmp_path / "maps"
    shutil.copytree(REFERENCE_MAPS, maps_path)

    maps = Maps(maps_path)
    maps.read()

    maps.training.splits[0].tmp.create_epoch(1)
    maps.training.splits[0].tmp.create_epoch(2)
    gen = maps.training.splits[0].tmp.iterdir()
    assert next(gen).path == (maps_path / "training" / "split-0" / "tmp" / "epoch-0")
    assert next(gen).path == (maps_path / "training" / "split-0" / "tmp" / "epoch-1")
    assert next(gen).path == (maps_path / "training" / "split-0" / "tmp" / "epoch-2")
    with pytest.raises(StopIteration):
        next(gen)


def test_get_checkpoint_path():
    maps = Maps(REFERENCE_MAPS)
    maps.read()
    models = maps.training.splits[0].models

    assert (
        models.get_checkpoint_dir("best-loss").path
        == models.best_models.metrics["loss"].path
    )
    with pytest.raises(
        KeyError,
        match=f"No checkpoint associated to the metric 'mse' in {str(models.best_models.path)}",
    ):
        models.get_checkpoint_dir("best-mse")

    assert (
        models.get_checkpoint_dir("epoch-0").path == models.checkpoints.epochs[0].path
    )
    with pytest.raises(
        KeyError,
        match=f"No checkpoint associated to epoch abc in {str(models.checkpoints.path)}",
    ):
        models.get_checkpoint_dir("epoch-abc")
    with pytest.raises(
        KeyError,
        match=f"No checkpoint associated to epoch 1 in {str(models.checkpoints.path)}",
    ):
        models.get_checkpoint_dir("epoch-1")

    assert models.get_checkpoint_dir("final").path == models.final.path

    with pytest.raises(
        ValueError,
        match="The name of the checkpoint must be like 'best-...', 'epoch-...' or 'final'. Got: abc",
    ):
        models.get_checkpoint_dir("abc")

    # models from split
    assert (
        maps.training.get_checkpoint_dir("split-0_best-loss").path
        == models.best_models.metrics["loss"].path
    )
    assert (
        maps.training.get_checkpoint_dir("split-0_epoch-0").path
        == models.checkpoints.epochs[0].path
    )
    assert maps.training.get_checkpoint_dir("split-0_final").path == models.final.path

    with pytest.raises(
        ValueError,
        match="The name of the checkpoint must be like 'split-..._best-...', 'split-..._epoch-...' or 'split-..._final'. Got: split-0-best-loss",
    ):
        maps.training.get_checkpoint_dir("split-0-best-loss")
    with pytest.raises(
        ValueError,
        match="The name of the checkpoint must be like 'split-..._best-...', 'split-..._epoch-...' or 'split-..._final'. Got: split-0_abc",
    ):
        maps.training.get_checkpoint_dir("split-0_abc")
    with pytest.raises(
        KeyError,
        match=f"No checkpoint associated to split 1 in {str(maps.training.path)}",
    ):
        maps.training.get_checkpoint_dir("split-1_best-loss")
    with pytest.raises(
        KeyError,
        match=f"No checkpoint associated to the metric 'mse' in {str(models.best_models.path)}",
    ):
        maps.training.get_checkpoint_dir("split-0_best-mse")


def test_read_checkpoint_name():
    maps = Maps(REFERENCE_MAPS)
    maps.read()

    assert maps.training.read_checkpoint_name("split-0_best-loss") == (0, "best-loss")
    assert maps.training.read_checkpoint_name("split-0_epoch-0") == (0, "epoch-0")
    assert maps.training.read_checkpoint_name("split-0_final") == (0, "final")
