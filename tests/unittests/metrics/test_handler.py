import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torchio as tio
from pydantic import ValidationError
from torch.nn import BCELoss

from clinicadl.data.dataloader.batch import Batch
from clinicadl.data.structures import DataPoint
from clinicadl.metrics import Metric
from clinicadl.metrics.config import LossMetricConfig, MSEMetricConfig
from clinicadl.metrics.handler import MetricsHandler
from clinicadl.utils.exceptions import (
    CannotReadJsonFieldError,
    ClinicaDLArgumentError,
    ClinicaDLConfigurationError,
)


class Model:
    def get_loss_functions(self):
        return {"loss": BCELoss()}


MODEL = Model()

DATAPOINTS = [
    DataPoint(
        image=tio.ScalarImage(tensor=torch.randn(1, 2, 2, 2)),
        participant=f"sub-{i}",
        session=f"ses-{i}",
        label=float(gt),
        output=float(pred),
    )
    for i, gt, pred in zip(range(6), [0, 0, 1, 1, 0, 1], [0, 1, 1, 0, 1, 1])
]

TSV_PATH = Path(__file__).parents[1] / "resources" / "metrics"

BATCH_1, BATCH_2 = Batch(DATAPOINTS[:3]), Batch(DATAPOINTS[3:])


class CustomMetric(Metric):
    _optimum = "max"

    def _accumulate(self, batch):
        pred = torch.tensor([datapoint["output"] for datapoint in batch])
        gt = torch.tensor([datapoint["label"] for datapoint in batch])
        return pred == gt

    def _aggregate(self, data):
        return data.float().mean().item()


def test_MetricsHandler():
    metrics = MetricsHandler(
        my_loss=LossMetricConfig(
            loss_name="loss",
        ),
        mse=MSEMetricConfig(),
        my_metric=CustomMetric(),
    )

    metrics.init_metrics(MODEL)
    output = metrics(BATCH_1)

    expected_df = pd.DataFrame.from_dict(
        {
            "participant_id": [f"sub-{i}" for i in range(3)],
            "session_id": [f"ses-{i}" for i in range(3)],
            "my_loss": pd.Series([0.0, 100.0, 0.0], dtype=np.float32),
            "mse": pd.Series([0.0, 1.0, 0.0], dtype=np.float32),
            "my_metric": pd.Series([1.0, 0.0, 1.0], dtype=np.float32),
        }
    )
    pd.testing.assert_frame_equal(output, expected_df)
    pd.testing.assert_frame_equal(metrics.detailed_df, expected_df)

    metrics(BATCH_2)
    expected_df = pd.DataFrame.from_dict(
        {
            "participant_id": [f"sub-{i}" for i in range(6)],
            "session_id": [f"ses-{i}" for i in range(6)],
            "my_loss": pd.Series(
                [0.0, 100.0, 0.0, 100.0, 100.0, 0.0], dtype=np.float32
            ),
            "mse": pd.Series([0.0, 1.0, 0.0, 1.0, 1.0, 0.0], dtype=np.float32),
            "my_metric": pd.Series([1.0, 0.0, 1.0, 0.0, 0.0, 1.0], dtype=np.float32),
        }
    )
    pd.testing.assert_frame_equal(metrics.detailed_df, expected_df)

    metrics.aggregate()
    expected_df = pd.DataFrame.from_dict(
        {
            "my_loss": pd.Series([50.0], dtype=np.float64),
            "mse": pd.Series([0.5], dtype=np.float64),
            "my_metric": pd.Series([0.5], dtype=np.float64),
        }
    )
    pd.testing.assert_frame_equal(metrics.df, expected_df)


def test_init_metrics():
    metrics = MetricsHandler(
        loss=LossMetricConfig(
            loss_name="loss_",
        ),
        mse=MSEMetricConfig(),
    )

    with pytest.raises(
        ClinicaDLArgumentError,
        match="In LossMetricConfig, loss_name='loss_' but there is no such loss*",
    ):
        metrics.init_metrics(MODEL)

    metrics = MetricsHandler(
        loss=LossMetricConfig(
            loss_name="loss",
        ),
        mse=MSEMetricConfig(),
    )

    with pytest.raises(
        ClinicaDLConfigurationError,
        match="First, call 'init_metrics' to instantiate the metrics.",
    ):
        metrics(BATCH_1)

    metrics.init_metrics(MODEL)
    assert len(metrics._metrics) == 2


def test_reset():
    metrics = MetricsHandler(
        mse=MSEMetricConfig(),
    )
    metrics.init_metrics(MODEL)

    metrics(BATCH_1)
    metrics.aggregate()
    metrics.reset(reset_df=True)
    assert len(metrics.df) == 0
    assert len(metrics.detailed_df) == 0
    metrics._metrics["mse"].get_buffer() is None

    metrics(BATCH_2)
    metrics.aggregate()
    metrics.reset()
    assert len(metrics.df) == 1
    assert len(metrics.detailed_df) == 3
    metrics._metrics["mse"].get_buffer() is None


def test_load(tmp_path):
    metrics = MetricsHandler(
        loss=LossMetricConfig(
            loss_fn=BCELoss(),
        ),
        my_metric=CustomMetric(),
        mse=MSEMetricConfig(),
    )
    metrics.init_metrics(MODEL)
    metrics.load(path=TSV_PATH / "validation.tsv")
    assert metrics._metrics["loss"].get_buffer() is None
    excepted_df = pd.DataFrame.from_dict(
        {
            "epoch": [0, 1, 2],
            "mse": [0.0, 0.1, 0.2],
            "my_metric": [1.0, 1.1, 1.2],
            "loss": [2.0, 2.1, 2.2],
        }
    )
    pd.testing.assert_frame_equal(metrics.df, excepted_df)

    metrics.load(
        path=TSV_PATH / "validation.tsv",
        details_path=TSV_PATH / "validation_details.tsv",
    )
    expected_details = pd.DataFrame.from_dict(
        {
            "epoch": [0, 0, 1, 1, 2, 2],
            "participant_id": [
                "sub-001",
                "sub-002",
                "sub-001",
                "sub-002",
                "sub-001",
                "sub-002",
            ],
            "session_id": [
                "ses-M000",
                "ses-M000",
                "ses-M000",
                "ses-M000",
                "ses-M000",
                "ses-M000",
            ],
            "mse": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
            "my_metric": [1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
            "loss": [2.0, 2.1, 2.2, 2.3, 2.4, 2.5],
        }
    )
    pd.testing.assert_frame_equal(metrics.df, excepted_df)
    pd.testing.assert_frame_equal(metrics.detailed_df, expected_details)

    metrics.reset(reset_df=True)
    metrics.save(tmp_path / "validation_empty.tsv")
    metrics.load(tmp_path / "validation_empty.tsv")

    with pytest.raises(
        AssertionError,
        match="Checkpoint in .* is not a valid metric file, some columns are missing: {'my_metric'}",
    ):
        metrics.load(TSV_PATH / "validation_bad.tsv")


def test_metrics_subset():
    metrics = MetricsHandler(
        mse=MSEMetricConfig(),
        my_metric=CustomMetric(),
    )
    metrics.init_metrics(MODEL)

    metrics(BATCH_1, epoch=0)
    metrics.aggregate(epoch=0)
    metrics.reset()

    with pytest.raises(
        ValueError,
        match=re.escape(
            "'abc' does not match any metrics. Metrics are: ['mse', 'my_metric']"
        ),
    ):
        output = metrics(BATCH_2, epoch=1, metrics=["abc"])

    output = metrics(BATCH_2, epoch=1, metrics=["mse"])
    metrics.aggregate(epoch=1, metrics=["mse"])

    expected_output_df = pd.DataFrame.from_dict(
        {
            "epoch": [1, 1, 1],
            "participant_id": [f"sub-{i}" for i in range(3, 6)],
            "session_id": [f"ses-{i}" for i in range(3, 6)],
            "mse": pd.Series([1.0, 1.0, 0.0], dtype=np.float32),
        }
    )
    expected_detailed_df = pd.DataFrame.from_dict(
        {
            "epoch": [0, 0, 0, 1, 1, 1],
            "participant_id": [f"sub-{i}" for i in range(6)],
            "session_id": [f"ses-{i}" for i in range(6)],
            "mse": pd.Series([0.0, 1.0, 0.0, 1.0, 1.0, 0.0], dtype=np.float32),
            "my_metric": pd.Series([1.0, 0.0, 1.0, -1, -1, -1], dtype=np.float32),
        }
    )
    expected_df = pd.DataFrame.from_dict(
        {
            "epoch": [0, 1],
            "mse": pd.Series([0.333333, 0.666666], dtype=np.float64),
            "my_metric": pd.Series([0.666666, -1], dtype=np.float64),
        }
    )
    pd.testing.assert_frame_equal(output, expected_output_df)
    pd.testing.assert_frame_equal(metrics.detailed_df.fillna(-1), expected_detailed_df)
    pd.testing.assert_frame_equal(metrics.df.fillna(-1), expected_df)


def test_epochs():
    metrics = MetricsHandler(
        mse=MSEMetricConfig(),
    )
    metrics.init_metrics(MODEL)

    # test on epochs
    metrics(BATCH_1)
    metrics.aggregate()
    metrics.reset()
    metrics(BATCH_2, epoch=0)
    metrics.aggregate(epoch=0)
    pd.testing.assert_series_equal(
        metrics.detailed_df["epoch"].fillna(-1),
        pd.Series([-1, -1, -1, 0, 0, 0], dtype=float, name="epoch"),
    )
    pd.testing.assert_series_equal(
        metrics.df["epoch"].fillna(-1), pd.Series([-1, 0], dtype=float, name="epoch")
    )


def test_add_metrics():
    metrics = MetricsHandler(
        mse=MSEMetricConfig(),
    )
    metrics.init_metrics(MODEL)
    metrics(BATCH_1)
    metrics.aggregate()
    metrics.add_metrics(my_metric=CustomMetric())
    metrics(BATCH_2)
    metrics.aggregate()
    expected_df = pd.DataFrame.from_dict(
        {
            "mse": pd.Series([0.33333, 0.66666], dtype=np.float64),
            "my_metric": pd.Series([np.nan, 0.33333], dtype=np.float64),
        }
    )
    pd.testing.assert_frame_equal(metrics.df, expected_df, rtol=1e-4)

    expected_df = pd.DataFrame.from_dict(
        {
            "participant_id": [f"sub-{i}" for i in range(6)],
            "session_id": [f"ses-{i}" for i in range(6)],
            "mse": pd.Series([0.0, 1.0, 0.0, 1.0, 1.0, 0.0], dtype=np.float32),
            "my_metric": pd.Series(
                [np.nan, np.nan, np.nan, 0.0, 0.0, 1.0], dtype=np.float32
            ),
        }
    )
    pd.testing.assert_frame_equal(metrics.detailed_df, expected_df)


def test_get_metric():
    metrics = MetricsHandler(
        mse=MSEMetricConfig(),
    )
    metrics.init_metrics(MODEL)
    metrics(BATCH_1)
    metrics.aggregate(epoch=0)
    metrics.reset()
    metrics(BATCH_2)
    metrics.aggregate(epoch=1)
    assert np.isclose(metrics.get_metric("mse"), 0.66666, rtol=1e-4)
    assert np.isclose(metrics.get_metric("mse", epoch=0), 0.33333, rtol=1e-4)


def test_checks():
    with pytest.raises(ValidationError):
        MetricsHandler(
            mse=lambda x: x,
        )

    config = MetricsHandler(
        mse=MSEMetricConfig(),
    )
    with pytest.raises(ValueError, match="A metric named 'mse' already exists!"):
        config.add_metrics(mse=MSEMetricConfig())
    with pytest.raises(ValidationError):
        config.add_metrics(my_metric=lambda x: x)


def test_save_and_merge_df(tmp_path):
    metrics = MetricsHandler(
        mse=MSEMetricConfig(),
    )
    metrics.init_metrics(MODEL)

    metrics(BATCH_1, epoch=0)
    metrics.aggregate(epoch=0)
    metrics.save(tmp_path / "df.tsv")
    df = pd.read_csv(tmp_path / "df.tsv", sep="\t")
    expected_df = pd.DataFrame.from_dict(
        {
            "epoch": [0],
            "mse": pd.Series([0.3333333]),
        }
    )
    pd.testing.assert_frame_equal(df, expected_df)

    metrics.save(tmp_path / "df.tsv", details_path=tmp_path / "detailed_df.tsv")
    df = pd.read_csv(tmp_path / "detailed_df.tsv", sep="\t")
    expected_df = pd.DataFrame.from_dict(
        {
            "epoch": [0, 0, 0],
            "participant_id": [f"sub-{i}" for i in range(3)],
            "session_id": [f"ses-{i}" for i in range(3)],
            "mse": pd.Series([0.0, 1.0, 0.0]),
        }
    )
    pd.testing.assert_frame_equal(df, expected_df)

    # merge
    metrics = MetricsHandler(
        my_metric=CustomMetric(),
    )
    metrics.init_metrics(MODEL)

    metrics(BATCH_1, epoch=0)
    metrics.aggregate(epoch=0)
    metrics.merge(tmp_path / "df.tsv", details_path=tmp_path / "detailed_df.tsv")

    df = pd.read_csv(tmp_path / "df.tsv", sep="\t")
    expected_df = pd.DataFrame.from_dict(
        {
            "epoch": [0],
            "mse": pd.Series([0.333333]),
            "my_metric": pd.Series([0.666666]),
        }
    )
    pd.testing.assert_frame_equal(df, expected_df)

    df = pd.read_csv(tmp_path / "detailed_df.tsv", sep="\t")
    expected_df = pd.DataFrame.from_dict(
        {
            "epoch": [0, 0, 0],
            "participant_id": [f"sub-{i}" for i in range(3)],
            "session_id": [f"ses-{i}" for i in range(3)],
            "mse": pd.Series([0.0, 1.0, 0.0]),
            "my_metric": pd.Series([1.0, 0.0, 1.0]),
        }
    )
    pd.testing.assert_frame_equal(df, expected_df)


def test_read_write_json(tmp_path):
    metrics = MetricsHandler(
        mse=MSEMetricConfig(),
    )
    metrics.add_metrics(my_metric=CustomMetric())
    metrics.to_json(tmp_path / "metrics.json")

    excepted_dict = {
        "name": "MetricsHandler",
        "metrics": {
            "mse": {
                "name": "MSEMetric",
                "get_not_nans": False,
                "pred_key": "output",
                "label_key": "label",
                "postprocessing": [],
                "reduction": "mean",
            },
            "my_metric": "CustomMetric",
        },
    }
    with open(tmp_path / "metrics.json", "r") as f:
        d = json.load(f)
    assert d == excepted_dict

    with pytest.raises(
        CannotReadJsonFieldError,
        match="MetricsHandler cannot read the field\\(s\\) \\['my_metric'\\] in .*\nPlease pass this field via kwargs.",
    ):
        MetricsHandler.from_json(json_path=tmp_path / "metrics.json")

    metrics = MetricsHandler.from_json(
        json_path=tmp_path / "metrics.json",
        my_metric=CustomMetric(),
    )
    assert isinstance(metrics.metrics["mse"], MSEMetricConfig)
    assert isinstance(metrics.metrics["my_metric"], CustomMetric)


def test_empty():
    metrics = MetricsHandler()
    metrics.init_metrics(MODEL)
    metrics(BATCH_1)
    empty_df = pd.DataFrame.from_dict(
        {
            "participant_id": [f"sub-{i}" for i in range(3)],
            "session_id": [f"ses-{i}" for i in range(3)],
        }
    )
    pd.testing.assert_frame_equal(metrics.detailed_df, empty_df)
