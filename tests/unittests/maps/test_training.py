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
)

from ..resources.objects import (
    CAPS_DIR,
    DATA,
    MAPS_DIR,
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
