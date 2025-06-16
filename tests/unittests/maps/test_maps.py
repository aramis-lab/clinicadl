from pathlib import Path
from typing import Dict

import pandas as pd
import pytest

from clinicadl.data.datasets import CapsDataset
from clinicadl.data.datatypes import PETLinear
from clinicadl.maps import Maps
from clinicadl.maps.data_group import DataGroup, TrainValDataGroup
from clinicadl.maps.split_dir.best_metric.best_metric import (
    BestMetric,
    BestMetricDataGroup,
)
from clinicadl.maps.split_dir.split_dir import SplitDir, TmpDir, TrainingLogs
from clinicadl.metrics.config.factory import MAEMetricConfig, MSEMetricConfig
from clinicadl.splitter import Split
from clinicadl.utils.exceptions import (
    ClinicaDLConfigurationError,
)

maps_test = Path(__file__).parents[1] / "resources" / "maps_test"
maps_example = Path(__file__).parents[1] / "resources" / "maps_example"
caps_dir = Path(__file__).parents[1] / "resources" / "caps_example"
data = pd.read_csv(caps_dir / "tsv" / "labels.tsv", sep="\t")

train_dataset = CapsDataset(
    caps_dir,
    preprocessing=PETLinear(
        tracer="18FAV45",
        suvr_reference_region="pons2",
        use_uncropped_image=True,
    ),
    data=data.iloc[:6],
)
val_dataset = CapsDataset(
    caps_dir,
    preprocessing=PETLinear(
        tracer="18FAV45",
        suvr_reference_region="pons2",
        use_uncropped_image=True,
    ),
    data=data.iloc[6:7],
)
test_dataset = CapsDataset(
    caps_dir,
    preprocessing=PETLinear(
        tracer="18FAV45",
        suvr_reference_region="pons2",
        use_uncropped_image=True,
    ),
    data=data.iloc[7:],
)

split_idx = 0
split = Split(
    index=split_idx,
    split_dir="abc",
    train_dataset=train_dataset,
    val_dataset=val_dataset,
)


def test_good_maps():
    maps = Maps(maps_test)
    if maps.exists():
        maps.remove()

    assert not maps.splits
    assert not maps.data_groups
    assert maps.groups_dir == maps_test / "groups"
    assert maps.train_val_tsv == maps_test / "train+validation.tsv"
    assert maps.requirements_txt == maps_test / "environment.txt"
    assert maps.computational_json == maps_test / "json" / "computational.json"
    assert maps.model_json == maps_test / "json" / "model.json"
    assert maps.optimization_json == maps_test / "json" / "optimization.json"
    assert maps.metrics_json == maps_test / "json" / "metrics.json"
    assert maps.exists() is False

    maps.create()

    assert maps.exists() is True
    assert maps.requirements_txt.is_file()
    assert maps.groups_dir.is_dir()

    assert len(maps.split_list) == 0
    assert len(maps.group_list) == 0

    maps.create_split(
        split=split, best_metrics=[MSEMetricConfig().name, MAEMetricConfig().name]
    )

    assert len(maps.split_list) == 1
    assert isinstance(maps.splits[split_idx], SplitDir)

    assert len(maps.data_groups) == 2
    assert "train" in maps.data_groups
    assert "validation" in maps.data_groups
    assert isinstance(maps.data_groups["train"], Dict)
    assert isinstance(maps.data_groups["validation"], Dict)
    assert isinstance(maps.data_groups["train"][split_idx], TrainValDataGroup)
    assert isinstance(maps.data_groups["validation"][split_idx], TrainValDataGroup)

    maps.create_data_group("test", dataset=test_dataset)

    assert len(maps.data_groups) == 3
    assert "test" in maps.data_groups
    assert isinstance(maps.data_groups["test"], DataGroup)

    assert len(maps.group_list) == 3
    assert "test" in maps.group_list

    maps.remove()


def test_good_split_dir():
    maps = Maps(maps_test)
    if maps.exists():
        maps.remove()
    maps.create()
    maps.create_split(
        split=split, best_metrics=[MSEMetricConfig().name, MAEMetricConfig().name]
    )

    assert (
        maps.splits[split_idx].split_json
        == maps_test / f"split-{split_idx}" / "split.json"
    )

    assert isinstance(maps.splits[split_idx].logs, TrainingLogs)
    assert (
        maps.splits[split_idx].logs.training_tsv
        == maps_test / f"split-{split_idx}" / "training_logs" / "training.tsv"
    )
    assert (
        maps.splits[split_idx].logs.tensorboard
        == maps_test / f"split-{split_idx}" / "training_logs" / "tensorboard"
    )

    assert isinstance(maps.splits[split_idx].tmp, TmpDir)
    assert (
        maps.splits[split_idx].tmp.checkpoint
        == maps_test / f"split-{split_idx}" / "tmp" / "checkpoint.pth.tar"
    )
    assert (
        maps.splits[split_idx].tmp.optimizer
        == maps_test / f"split-{split_idx}" / "tmp" / "optimizer.pth.tar"
    )

    maps.remove()


def test_good_best_metrics():
    maps = Maps(maps_test)
    if maps.exists():
        maps.remove()
    maps.create()
    maps.create_split(
        split=split, best_metrics=[MSEMetricConfig().name, MAEMetricConfig().name]
    )

    assert len(maps.splits[split_idx].best_metrics) == 2

    assert "MSEMetric" in maps.splits[split_idx].best_metrics.keys()
    assert "MAEMetric" in maps.splits[split_idx].best_metrics.keys()

    assert isinstance(maps.splits[split_idx].best_metrics["MSEMetric"], BestMetric)
    assert isinstance(maps.splits[split_idx].best_metrics["MAEMetric"], BestMetric)

    assert (
        maps.splits[split_idx].best_metrics["MSEMetric"].metric
        == MSEMetricConfig().name
    )
    assert isinstance(
        maps.splits[split_idx].best_metrics["MSEMetric"].train, BestMetricDataGroup
    )
    assert isinstance(
        maps.splits[split_idx].best_metrics["MSEMetric"].val, BestMetricDataGroup
    )
    assert len(maps.splits[split_idx].best_metrics["MSEMetric"].data_groups) == 0
    assert (
        maps.splits[split_idx].best_metrics["MSEMetric"].model
        == maps_test / f"split-{split_idx}" / "best-MSEMetric" / "model.pth.tar"
    )

    maps.splits[split_idx].best_metrics["MSEMetric"].create_data_group(name="test")

    assert len(maps.splits[split_idx].best_metrics["MSEMetric"].data_groups) == 1
    assert (
        maps.splits[split_idx].best_metrics["MSEMetric"].data_groups["test"].name
        == "test"
    )
    assert (
        maps.splits[split_idx]
        .best_metrics["MSEMetric"]
        .data_groups["test"]
        .description_log
        == maps_test
        / f"split-{split_idx}"
        / "best-MSEMetric"
        / "test"
        / "description.log"
    )
    assert (
        maps.splits[split_idx]
        .best_metrics["MSEMetric"]
        .data_groups["test"]
        .predictions_tsv
        == maps_test
        / f"split-{split_idx}"
        / "best-MSEMetric"
        / "test"
        / "predictions.tsv"
    )
    assert (
        maps.splits[split_idx].best_metrics["MSEMetric"].data_groups["test"].metrics_tsv
        == maps_test / f"split-{split_idx}" / "best-MSEMetric" / "test" / "metrics.tsv"
    )
    assert (
        maps.splits[split_idx].best_metrics["MSEMetric"].data_groups["test"].caps_output
        == maps_test / f"split-{split_idx}" / "best-MSEMetric" / "test" / "CAPSOutput"
    )

    maps.remove()


def test_load_maps():
    maps = Maps(maps_example)
    maps.load()
    assert maps.exists()
    assert maps.requirements_txt.is_file()
    assert maps.groups_dir.is_dir()
    assert maps.split_list == [0]
    assert maps.group_list.sort() == ["train", "validation", "test"].sort()
    assert len(maps.splits) == 1
    assert isinstance(maps.splits[0], SplitDir)
    assert maps.splits[0].split_json == maps_example / "split-0" / "split.json"
    assert maps.splits[0].split_json.is_file()
    assert len(maps.splits[0].best_metrics_list) == 2
    assert isinstance(maps.splits[0].best_metrics["MSEMetric"], BestMetric)
    assert isinstance(maps.splits[0].best_metrics["Loss"], BestMetric)
    assert isinstance(
        maps.splits[0].best_metrics["MSEMetric"].train, BestMetricDataGroup
    )
    assert isinstance(maps.splits[0].best_metrics["MSEMetric"].val, BestMetricDataGroup)
    assert isinstance(maps.splits[0].best_metrics["Loss"].train, BestMetricDataGroup)
    assert isinstance(maps.splits[0].best_metrics["Loss"].val, BestMetricDataGroup)
    assert len(maps.splits[0].best_metrics["MSEMetric"].data_groups) == 2
    assert maps.splits[0].best_metrics["MSEMetric"].data_groups["train"].name == "train"
    assert (
        maps.splits[0].best_metrics["MSEMetric"].data_groups["train"].description_log
        == maps_example / "split-0" / "best-MSEMetric" / "train" / "description.log"
    )
    assert (
        maps.splits[0].best_metrics["MSEMetric"].data_groups["train"].predictions_tsv
        == maps_example / "split-0" / "best-MSEMetric" / "train" / "predictions.tsv"
    )

    assert len(maps.data_groups) == 3
    assert isinstance(maps.data_groups["train"], Dict)
    assert isinstance(maps.data_groups["validation"], Dict)
    assert isinstance(maps.data_groups["test"], DataGroup)
    assert isinstance(maps.data_groups["train"][0], TrainValDataGroup)
    assert isinstance(maps.data_groups["validation"][0], TrainValDataGroup)
    assert isinstance(maps.data_groups["test"], DataGroup)
    assert maps.train_val_tsv == maps_example / "train+validation.tsv"


def test_bad_load():
    maps = Maps("false_maps")

    with pytest.raises(ClinicaDLConfigurationError):
        maps.load()

    maps = Maps(maps_example)
    (maps_example / "split-1").mkdir()
    with pytest.raises(ClinicaDLConfigurationError):
        maps.load()

    (maps_example / "split-1").rmdir()


def test_bad_best_metrics():
    maps = Maps(maps_test)
    if maps.exists():
        maps.remove()
    maps.create()
    maps.create_split(
        split=split, best_metrics=[MSEMetricConfig().name, MAEMetricConfig().name]
    )

    with pytest.raises(ClinicaDLConfigurationError):
        maps.splits[split_idx].best_metrics["MSEMetric"].create(split)

    maps.splits[split_idx].best_metrics["MSEMetric"].create_data_group(name="test")

    with pytest.raises(ClinicaDLConfigurationError):
        maps.splits[split_idx].best_metrics["MSEMetric"].create_data_group(name="test")

    maps.remove()


def test_bad_split_dir():
    maps = Maps(maps_test)
    if maps.exists():
        maps.remove()
    maps.create()
    maps.create_split(
        split=split, best_metrics=[MSEMetricConfig().name, MAEMetricConfig().name]
    )

    with pytest.raises(ClinicaDLConfigurationError):
        maps.splits[split_idx].create(split)

    maps.remove()


def test_bad_maps():
    maps = Maps("maps_test")
    if maps.exists():
        maps.remove()

    with pytest.raises(ClinicaDLConfigurationError):
        maps.split_list

    with pytest.raises(ClinicaDLConfigurationError):
        maps.group_list

    maps.create()
    with pytest.raises(ClinicaDLConfigurationError):
        maps.create()

    maps.create_split(
        split=split, best_metrics=[MSEMetricConfig().name, MAEMetricConfig().name]
    )

    with pytest.raises(ClinicaDLConfigurationError):
        maps.create_split(
            split=split, best_metrics=[MSEMetricConfig().name, MAEMetricConfig().name]
        )

    maps.create_data_group("test", dataset=test_dataset)

    with pytest.raises(ClinicaDLConfigurationError):
        maps.create_data_group("test", dataset=test_dataset)

    maps.remove()
