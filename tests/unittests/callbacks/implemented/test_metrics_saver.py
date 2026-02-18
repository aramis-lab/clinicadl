from pathlib import Path
from unittest.mock import Mock

from clinicadl.callbacks.implemented import MetricsSaverCallback
from clinicadl.io import Maps

MAPS_PATH = Path(__file__).parents[2] / "resources" / "maps_example"
MAPS = Maps(MAPS_PATH)
MAPS.read()
STATE = Mock()
STATE.split_idx = 0


def test_train():
    METRICS = Mock()
    saver = MetricsSaverCallback()
    saver.on_validation_end(metrics=METRICS)
    MAPS.training.splits[0].validation_metrics.create = Mock()
    saver.on_train_end(state=STATE, maps=MAPS)
    MAPS.training.splits[0].validation_metrics.create.assert_called()

    METRICS.save.assert_called_once_with(
        path=MAPS.training.splits[0].validation_metrics.aggregated_tsv,
        details_path=MAPS.training.splits[0].validation_metrics.details_tsv,
    )


def test_validate():
    STATE.split_idx = 0
    METRICS = Mock()
    saver = MetricsSaverCallback()
    saver.on_validate_start(model_checkpoint="best-loss")
    saver.on_validate_end(state=STATE, maps=MAPS, metrics=METRICS)

    METRICS.merge.assert_called_once_with(
        path=MAPS.training.splits[0]
        .models.best_models.metrics["loss"]
        .validation_metrics.aggregated_tsv,
        details_path=MAPS.training.splits[0]
        .models.best_models.metrics["loss"]
        .validation_metrics.details_tsv,
    )


def test_test():
    STATE.split_idx = 0
    METRICS = Mock()
    saver = MetricsSaverCallback()
    saver.on_test_start(model_checkpoint="split-0_best-loss", group_name="X")
    saver.on_test_end(maps=MAPS, metrics=METRICS)

    METRICS.save.assert_called_once_with(
        path=MAPS.test.groups["X"]
        .results.splits[0]
        .models["best-loss"]
        .metrics.aggregated_tsv,
        details_path=MAPS.test.groups["X"]
        .results.splits[0]
        .models["best-loss"]
        .metrics.details_tsv,
    )
