import shutil
from pathlib import Path

import pytest

from clinicadl.io.maps import Maps

REFERENCE_MAPS = Path(__file__).parents[1] / "resources" / "maps_example"


def test_maps(tmp_path: Path):
    maps_path = tmp_path / "maps"
    maps = Maps(maps_path)

    maps.create()
    assert (maps_path / "summary.log").is_file()
    assert (maps_path / "environment.txt").is_file()
    assert maps.training.data.train.is_empty()

    assert maps.architecture_log == (maps_path / "architecture.log")
    assert maps.model_json == (maps_path / "model.json")
    assert maps.torchsummary_txt == (maps_path / "torchsummary.txt")

    # training
    assert (maps_path / "training").exists()

    assert maps.training.metrics_json == (maps_path / "training" / "metrics.json")
    assert maps.training.optimization_json == (
        maps_path / "training" / "optimization.json"
    )
    assert maps.training.callbacks_json == (maps_path / "training" / "callbacks.json")

    # training - split
    maps.training.create_split(0)
    assert maps.training.splits_list == [0]

    assert maps.training.splits[0].dataset_json == (
        maps_path / "training" / "split-0" / "dataset.json"
    )
    assert maps.training.splits[0].dataloader_json == (
        maps_path / "training" / "split-0" / "dataloader.json"
    )
    assert maps.training.splits[0].computational_json == (
        maps_path / "training" / "split-0" / "computational.json"
    )
    assert maps.training.splits[0].performance_txt == (
        maps_path / "training" / "split-0" / "performance.txt"
    )

    # training - split - validation_metrics
    assert (maps_path / "training" / "split-0" / "validation_metrics").exists()
    assert maps.training.splits[0].validation_metrics.aggregated == (
        maps_path / "training" / "split-0" / "validation_metrics" / "aggregated.tsv"
    )
    assert maps.training.splits[0].validation_metrics.details == (
        maps_path / "training" / "split-0" / "validation_metrics" / "details.tsv"
    )

    # training - split - logs
    assert (maps_path / "training" / "split-0" / "logs").exists()
    assert maps.training.splits[0].logs.training_loss == (
        maps_path / "training" / "split-0" / "logs" / "training_loss.tsv"
    )
    assert maps.training.splits[0].logs.tensorboard.path == (
        maps_path / "training" / "split-0" / "logs" / "tensorboard"
    )
    assert (maps_path / "training" / "split-0" / "logs" / "tensorboard").exists()

    # training - split - checkpoints
    assert (maps_path / "training" / "split-0" / "checkpoints").exists()
    maps.training.splits[0].checkpoints.create_epoch(0)
    assert maps.training.splits[0].checkpoints.epochs_list == [0]
    assert (maps_path / "training" / "split-0" / "checkpoints" / "epoch-0").exists()

    assert maps.training.splits[0].checkpoints.epochs[0].model == (
        maps_path / "training" / "split-0" / "checkpoints" / "epoch-0" / "model.pth.tar"
    )

    # training - split - tmp
    assert (maps_path / "training" / "split-0" / "tmp").exists()
    maps.training.splits[0].tmp.create_epoch(0)
    assert maps.training.splits[0].tmp.epochs_list == [0]
    assert (maps_path / "training" / "split-0" / "tmp" / "epoch-0").exists()

    assert maps.training.splits[0].tmp.epochs[0].callbacks.path == (
        maps_path / "training" / "split-0" / "tmp" / "epoch-0" / "callbacks"
    )
    assert (
        maps_path / "training" / "split-0" / "tmp" / "epoch-0" / "callbacks"
    ).exists()
    assert maps.training.splits[0].tmp.epochs[0].stop == (
        maps_path / "training" / "split-0" / "tmp" / "epoch-0" / "stop.json"
    )
    assert maps.training.splits[0].tmp.epochs[0].model == (
        maps_path / "training" / "split-0" / "tmp" / "epoch-0" / "model.pth.tar"
    )

    # training - split - tmp - validation_metrics
    assert (
        maps_path / "training" / "split-0" / "tmp" / "epoch-0" / "validation_metrics"
    ).exists()
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

    # training - split - best_models
    maps.training.splits[0].create_metric("loss")
    assert maps.training.splits[0].metrics_list == ["loss"]
    assert (maps_path / "training" / "split-0" / "best-model-loss").exists()
    assert maps.training.splits[0].best_models["loss"].model == (
        maps_path / "training" / "split-0" / "best-model-loss" / "model.pth.tar"
    )

    # training - split - best_models - validation_metrics
    assert (
        maps_path / "training" / "split-0" / "best-model-loss" / "validation_metrics"
    ).exists()
    assert maps.training.splits[0].best_models[
        "loss"
    ].validation_metrics.aggregated == (
        maps_path
        / "training"
        / "split-0"
        / "best-model-loss"
        / "validation_metrics"
        / "aggregated.tsv"
    )
    assert maps.training.splits[0].best_models["loss"].validation_metrics.details == (
        maps_path
        / "training"
        / "split-0"
        / "best-model-loss"
        / "validation_metrics"
        / "details.tsv"
    )

    # training - data
    assert maps.training.data.data_tsv == (maps_path / "training" / "data" / "data.tsv")

    # training - data - train
    assert (maps_path / "training" / "data" / "train").exists()
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
    assert (maps_path / "training" / "data" / "validation").exists()
    assert maps.training.data.validation.dataset_json == (
        maps_path / "training" / "data" / "validation" / "dataset.json"
    )

    maps.training.data.validation.create_split(0)
    assert maps.training.data.validation.splits_list == [0]
    assert (maps_path / "training" / "data" / "validation" / "split-0").exists()
    assert maps.training.data.validation.splits[0].data_tsv == (
        maps_path / "training" / "data" / "validation" / "split-0" / "data.tsv"
    )

    # predictions
    assert (maps_path / "predictions").exists()

    maps.predictions.create_group("X")
    assert maps.predictions.groups_list == ["X"]
    assert (maps_path / "predictions" / "group-X").exists()
    assert maps.predictions.groups["X"]
    assert maps.predictions.groups["X"].data_tsv == (
        maps_path / "predictions" / "group-X" / "data.tsv"
    )
    assert maps.predictions.groups["X"].dataset_json == (
        maps_path / "predictions" / "group-X" / "dataset.json"
    )

    # predictions - group - split
    maps.predictions.groups["X"].create_split(0)
    assert maps.predictions.groups["X"].splits_list == [0]
    assert (maps_path / "predictions" / "group-X" / "split-0").exists()

    # predictions - group - split - best_models
    maps.predictions.groups["X"].splits[0].create_metric("loss")
    assert maps.predictions.groups["X"].splits[0].metrics_list == ["loss"]
    assert (
        maps_path
        / "predictions"
        / "group-X"
        / "split-0"
        / "best-model-loss"
        / "metrics"
    ).exists()
    assert maps.predictions.groups["X"].splits[0].best_models[
        "loss"
    ].metrics.aggregated == (
        maps_path
        / "predictions"
        / "group-X"
        / "split-0"
        / "best-model-loss"
        / "metrics"
        / "aggregated.tsv"
    )
    assert maps.predictions.groups["X"].splits[0].best_models[
        "loss"
    ].metrics.details == (
        maps_path
        / "predictions"
        / "group-X"
        / "split-0"
        / "best-model-loss"
        / "metrics"
        / "details.tsv"
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

    maps.training.splits[0].best_models["loss"].validation_metrics.aggregated.unlink()
    with pytest.raises(FileNotFoundError, match="A directory or a file is missing: .*"):
        maps.read()

    maps = Maps(tmp_path / "maps_bis")
    with pytest.raises(FileNotFoundError, match="Directory .* does not exist."):
        maps.read()
