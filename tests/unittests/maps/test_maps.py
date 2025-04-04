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
    ClinicaDLArgumentError,
    ClinicaDLCAPSError,
    ClinicaDLConfigurationError,
)

maps_path = Path(__file__).parents[1] / "resources" / "maps_test"
caps_dir = Path(__file__).parents[1] / "resources" / "caps_example"
data = pd.read_csv(caps_dir / "labels.tsv", sep="\t")

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
    maps = Maps(maps_path)
    maps.remove()

    assert maps.splits == {}
    assert maps.data_groups == {}
    assert maps.groups_dir == maps_path / "groups"
    assert maps.train_val_tsv == maps_path / "train+validation.tsv"
    assert maps.requirements_txt == maps_path / "environment.txt"
    assert maps.computational_json == maps_path / "JSON" / "computational.json"
    assert maps.model_json == maps_path / "JSON" / "model.json"
    assert maps.optimization_json == maps_path / "JSON" / "optimization.json"
    assert maps.metrics_json == maps_path / "JSON" / "metrics.json"
    assert maps.exists() is False

    maps.create()

    assert maps.exists() is True
    assert maps.requirements_txt.is_file()
    assert maps.groups_dir.is_dir()

    assert len(maps.split_list) == 0
    assert len(maps.group_list) == 0

    maps.create_split(split=split, best_metrics=[MSEMetricConfig(), MAEMetricConfig()])

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

    assert len(maps.group_list) == 1
    assert maps.group_list[0] == "test"

    maps.remove()


def test_good_split_dir():
    maps = Maps(maps_path)
    maps.remove()
    maps.create()
    maps.create_split(split=split, best_metrics=[MSEMetricConfig(), MAEMetricConfig()])

    assert (
        maps.splits[split_idx].split_json
        == maps_path / f"split-{split_idx}" / "split.json"
    )

    assert isinstance(maps.splits[split_idx].logs, TrainingLogs)
    assert (
        maps.splits[split_idx].logs.training_tsv
        == maps_path / f"split-{split_idx}" / "training_logs" / "training.tsv"
    )
    assert (
        maps.splits[split_idx].logs.tensorboard
        == maps_path / f"split-{split_idx}" / "training_logs" / "tensorboard"
    )

    assert isinstance(maps.splits[split_idx].tmp, TmpDir)
    assert (
        maps.splits[split_idx].tmp.checkpoint
        == maps_path / f"split-{split_idx}" / "tmp" / "checkpoint.pth.tar"
    )
    assert (
        maps.splits[split_idx].tmp.optimizer
        == maps_path / f"split-{split_idx}" / "tmp" / "optimizer.pth.tar"
    )

    maps.remove()


def test_good_best_metrics():
    maps = Maps(maps_path)
    maps.remove()
    maps.create()
    maps.create_split(split=split, best_metrics=[MSEMetricConfig(), MAEMetricConfig()])

    assert len(maps.splits[split_idx].best_metrics) == 2

    assert "MSEMetric" in maps.splits[split_idx].best_metrics.keys()
    assert "MAEMetric" in maps.splits[split_idx].best_metrics.keys()

    assert isinstance(maps.splits[split_idx].best_metrics["MSEMetric"], BestMetric)
    assert isinstance(maps.splits[split_idx].best_metrics["MAEMetric"], BestMetric)

    assert maps.splits[split_idx].best_metrics["MSEMetric"].metric == MSEMetricConfig()
    assert isinstance(
        maps.splits[split_idx].best_metrics["MSEMetric"].train, BestMetricDataGroup
    )
    assert isinstance(
        maps.splits[split_idx].best_metrics["MSEMetric"].val, BestMetricDataGroup
    )
    assert len(maps.splits[split_idx].best_metrics["MSEMetric"].data_groups) == 0
    assert (
        maps.splits[split_idx].best_metrics["MSEMetric"].model
        == maps_path / f"split-{split_idx}" / "best-MSEMetric" / "model.pth.tar"
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
        == maps_path
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
        == maps_path
        / f"split-{split_idx}"
        / "best-MSEMetric"
        / "test"
        / "predictions.tsv"
    )
    assert (
        maps.splits[split_idx].best_metrics["MSEMetric"].data_groups["test"].metrics_tsv
        == maps_path / f"split-{split_idx}" / "best-MSEMetric" / "test" / "metrics.tsv"
    )
    assert (
        maps.splits[split_idx].best_metrics["MSEMetric"].data_groups["test"].caps_output
        == maps_path / f"split-{split_idx}" / "best-MSEMetric" / "test" / "CAPSOutput"
    )

    maps.remove()


def test_bad_best_metrics():
    maps = Maps(maps_path)
    maps.remove()
    maps.create()
    maps.create_split(split=split, best_metrics=[MSEMetricConfig(), MAEMetricConfig()])

    with pytest.raises(ClinicaDLConfigurationError):
        maps.splits[split_idx].best_metrics["MSEMetric"].create(split)

    maps.splits[split_idx].best_metrics["MSEMetric"].create_data_group(name="test")

    with pytest.raises(ClinicaDLConfigurationError):
        maps.splits[split_idx].best_metrics["MSEMetric"].create_data_group(name="test")

    maps.remove()


def test_bad_split_dir():
    maps = Maps(maps_path)
    maps.remove()
    maps.create()
    maps.create_split(split=split, best_metrics=[MSEMetricConfig(), MAEMetricConfig()])

    with pytest.raises(ClinicaDLConfigurationError):
        maps.splits[split_idx].create(split)

    maps.remove()


def test_bad_maps():
    maps = Maps("maps_test")
    maps.remove()

    with pytest.raises(ClinicaDLConfigurationError):
        maps.split_list

    with pytest.raises(ClinicaDLConfigurationError):
        maps.group_list

    maps.create()
    with pytest.raises(ClinicaDLConfigurationError):
        maps.create()

    maps.create_split(split=split, best_metrics=[MSEMetricConfig(), MAEMetricConfig()])

    with pytest.raises(ClinicaDLConfigurationError):
        maps.create_split(
            split=split, best_metrics=[MSEMetricConfig(), MAEMetricConfig()]
        )

    maps.create_data_group("test", dataset=test_dataset)

    with pytest.raises(ClinicaDLConfigurationError):
        maps.create_data_group("test", dataset=test_dataset)

    maps.remove()
