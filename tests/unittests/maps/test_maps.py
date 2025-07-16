from copy import deepcopy
from pathlib import Path
from typing import Dict

import pandas as pd
import pytest

from clinicadl.data.datasets import CapsDataset
from clinicadl.data.datatypes import PETLinear
from clinicadl.maps import Maps
from clinicadl.maps.training import TrainingDir
from clinicadl.maps.training.data.data import DataDir, DataTrainValDir
from clinicadl.maps.training.data.splits import DataSplitDir
from clinicadl.maps.training.splits.best_metrics import TrainBestMetricDir
from clinicadl.maps.training.splits.checkpoints import CheckpointsDir
from clinicadl.maps.training.splits.logs import LogsDir
from clinicadl.maps.training.splits.splits import TrainSplitDir
from clinicadl.maps.training.splits.tmp import TmpDir
from clinicadl.metrics.config.factory import MAEMetricConfig, MSEMetricConfig
from clinicadl.split import Split
from clinicadl.utils.exceptions import (
    ClinicaDLConfigurationError,
    ClinicaDLTestingError,
)

from ..resources.objects import (
    CAPS_DIR,
    COMP,
    DATA,
    MAPS_DIR,
    METRICS,
    MODEL,
    OPTIM,
    SPLIT,
    TEST_DATASET,
    TRAIN_DATASET,
    VAL_DATASET,
)

testing_maps = Path(__file__).parents[1] / "resources" / "maps_test"


def test_good_maps():
    maps = Maps(testing_maps)
    if maps.exists():
        maps.remove()

    assert not maps.training.splits

    assert maps.training.data.df is None
    assert not maps.training.data.val.splits
    assert not maps.training.data.train.splits

    assert maps.path == testing_maps
    assert maps.training.path == testing_maps / "training"

    assert maps.training.data.path == testing_maps / "training" / "data"
    assert maps.training.data.train.path == testing_maps / "training" / "data" / "train"
    assert (
        maps.training.data.val.path == testing_maps / "training" / "data" / "validation"
    )

    assert (
        maps.training.computational_json
        == testing_maps / "training" / "computational.json"
    )
    assert maps.training.metrics_json == testing_maps / "training" / "metrics.json"
    assert (
        maps.training.optimization_json
        == testing_maps / "training" / "optimization.json"
    )
    assert maps.training.callbacks_json == testing_maps / "training" / "callbacks.json"

    assert maps.architecture_log == testing_maps / "architecture.log"
    assert maps.environment_txt == testing_maps / "environment.txt"
    assert maps.model_json == testing_maps / "model.json"
    assert maps.summary_log == testing_maps / "summary.log"

    assert not maps.exists()

    maps._create_dirs()

    assert maps.exists()
    assert maps.environment_txt.is_file()
    assert maps.training.path.is_dir()
    assert maps.predictions.path.is_dir()

    assert maps.training.split_list == []
    assert maps.training.data.df is None

    assert not maps.training.data.val.splits
    assert not maps.training.data.train.splits

    maps._create_training_split(split=SPLIT)

    assert len(maps.training.split_list) == 1
    assert maps.training.split_list == [SPLIT.index]
    assert (
        maps.training.splits[SPLIT.index].path
        == TrainSplitDir(num=SPLIT.index, parents_path=maps.training.path).path
    )
    assert maps.training.splits[SPLIT.index].best_metrics_list == []

    assert maps.training.splits[SPLIT.index].logs.path.is_dir()
    assert maps.training.splits[SPLIT.index].tmp.path.is_dir()
    assert maps.training.splits[SPLIT.index].checkpoints.path.is_dir()

    assert maps.training.data.train.split_list == [SPLIT.index]
    assert maps.training.data.train.splits[SPLIT.index].data_tsv.is_file()

    assert maps.training.data.val.split_list == [SPLIT.index]
    assert maps.training.data.val.splits[SPLIT.index].data_tsv.is_file()

    maps.training.splits[SPLIT.index]._create_best_metrics(metric="mse")

    assert len(maps.training.splits[SPLIT.index].best_metrics_list) == 1
    assert maps.training.splits[SPLIT.index].best_metrics["mse"].path.is_dir()

    maps.remove()


def test_load_maps_training():
    maps = Maps(MAPS_DIR)

    if not maps.exists():
        raise ClinicaDLTestingError(
            "MAPS directory does not exist. Please add a valid MAPS directory."
        )

    maps.load()

    assert maps.training.exists()

    assert maps.training.data.exists()
    assert maps.training.data.df is not None

    assert maps.training.data.train.exists()
    assert maps.training.data.train.split_list == [0, 1, 2]
    for split in maps.training.data.train.split_list:
        assert maps.training.data.train.splits[split].exists()
        assert maps.training.data.train.splits[split].df is not None
        assert maps.training.data.train.splits[split].data_tsv.is_file()
        assert isinstance(maps.training.data.train.splits[split], DataSplitDir)

    assert maps.training.data.val.exists()
    assert maps.training.data.val.split_list == [0, 1, 2]
    for split in maps.training.data.val.split_list:
        assert maps.training.data.val.splits[split].exists()
        assert maps.training.data.val.splits[split].df is not None
        assert maps.training.data.val.splits[split].data_tsv.is_file()
        assert isinstance(maps.training.data.val.splits[split], DataSplitDir)

    assert maps.training.splits
    assert maps.training.splits[0].exists()

    assert maps.training.splits[0].best_metrics_list == ["loss", "mae"]
    assert maps.training.splits[0].best_metrics["loss"].exists()
    assert maps.training.splits[0].best_metrics["mae"].exists()

    assert maps.training.splits[0].logs.exists()
    assert maps.training.splits[0].logs.training_tsv.is_file()

    assert maps.training.splits[0].checkpoints.exists()
    assert maps.training.splits[0].checkpoints.epochs[10].exists()
    assert maps.training.splits[0].checkpoints.epochs[42].exists()

    assert maps.training.splits[0].summary_log.is_file()
    assert maps.training.splits[0].validation_metrics_tsv.is_file()

    assert maps.training.callbacks_json.is_file()
    assert maps.training.computational_json.is_file()
    assert maps.training.optimization_json.is_file()
    assert maps.training.metrics_json.is_file()

    assert maps.architecture_log.is_file()
    assert maps.environment_txt.is_file()
    assert maps.model_json.is_file()
    assert maps.summary_log.is_file()


def test_load_maps_predictions():
    maps = Maps(MAPS_DIR)

    if not maps.exists():
        raise ClinicaDLTestingError(
            "MAPS directory does not exist. Please add a valid MAPS directory."
        )

    maps.load()

    assert maps.predictions.exists()

    assert maps.predictions.groups
    assert maps.predictions.group_list == ["ADNI", "OASIS"]

    assert maps.predictions.groups["ADNI"].exists()
    assert maps.predictions.groups["ADNI"].caps_dataset_json.is_file()
    assert maps.predictions.groups["ADNI"].data_tsv.is_file()
    assert maps.predictions.groups["ADNI"].metrics_json.is_file()

    assert maps.predictions.groups["ADNI"].split_list == [0, 1]
    assert maps.predictions.groups["ADNI"].splits[0].exists()
    assert maps.predictions.groups["ADNI"].splits[0].best_metrics["loss"].exists()
    assert maps.predictions.groups["ADNI"].splits[0].best_metrics["mae"].exists()

    assert (
        maps.predictions.groups["ADNI"]
        .splits[0]
        .best_metrics["loss"]
        .metrics_tsv.is_file()
    )
    assert (
        maps.predictions.groups["ADNI"]
        .splits[0]
        .best_metrics["loss"]
        .caps_output.exists()
    )

    assert maps.predictions.groups["ADNI"].splits[0].computational_json.is_file()


def test_maps_training_data():
    maps = Maps(MAPS_DIR)

    if not maps.exists():
        raise ClinicaDLTestingError(
            "MAPS directory does not exist. Please add a valid MAPS directory."
        )

    maps.load()

    assert maps.training.get_computational_config() == COMP
    assert maps.training.get_optimization_config() == OPTIM

    m1 = str(deepcopy(maps.training.get_metrics()))
    m2 = str(deepcopy(METRICS.metrics))
    assert m1 == m2

    n1 = str(deepcopy(maps.get_model().network))
    n2 = str(deepcopy(MODEL.network))
    assert n1 == n2

    l1 = str(deepcopy(maps.get_model().loss))
    l2 = str(deepcopy(MODEL.loss))
    assert l1 == l2

    o1 = str(deepcopy(maps.get_model().optimizer))
    o2 = str(deepcopy(MODEL.optimizer))
    assert o1 == o2


def test_bad_maps():
    maps = Maps(testing_maps)

    if maps.exists():
        maps.remove()

    with pytest.raises(ClinicaDLConfigurationError):
        maps.load()

    my_maps = Maps(MAPS_DIR)
    my_maps.load()
