import json

import numpy as np
import pandas as pd
import pytest
import torch
import torchio as tio
from pydantic import ValidationError
from torch.nn import BCELoss

from clinicadl.data.dataloader.batch import SimpleBatch
from clinicadl.data.structures import DataPoint
from clinicadl.metrics import Metric
from clinicadl.metrics.config import LossMetricConfig, MSEMetricConfig
from clinicadl.metrics.handler import MetricsHandler

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

BATCH_1, BATCH_2 = SimpleBatch(DATAPOINTS[:3]), SimpleBatch(DATAPOINTS[3:])


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
        loss=LossMetricConfig(
            loss_fn=BCELoss(),
        ),
        mse=MSEMetricConfig(),
        my_metric=CustomMetric(),
    )

    metrics(BATCH_1)
    expected_df = pd.DataFrame.from_dict(
        {
            "mse": pd.Series([0.0, 1.0, 0.0], dtype=np.float32),
            "my_metric": pd.Series([1.0, 0.0, 1.0], dtype=np.float32),
            "loss": pd.Series([0.0, 100.0, 0.0], dtype=np.float32),
            "participant_id": [f"sub-{i}" for i in range(3)],
            "session_id": [f"ses-{i}" for i in range(3)],
        }
    )
    pd.testing.assert_frame_equal(metrics.detailed_df, expected_df)

    metrics(BATCH_2)
    expected_df = pd.DataFrame.from_dict(
        {
            "mse": pd.Series([0.0, 1.0, 0.0, 1.0, 1.0, 0.0], dtype=np.float32),
            "my_metric": pd.Series([1.0, 0.0, 1.0, 0.0, 0.0, 1.0], dtype=np.float32),
            "loss": pd.Series([0.0, 100.0, 0.0, 100.0, 100.0, 0.0], dtype=np.float32),
            "participant_id": [f"sub-{i}" for i in range(6)],
            "session_id": [f"ses-{i}" for i in range(6)],
        }
    )
    pd.testing.assert_frame_equal(metrics.detailed_df, expected_df)

    metrics.aggregate()
    expected_df = pd.DataFrame.from_dict(
        {
            "mse": pd.Series([0.5], dtype=np.float64),
            "my_metric": pd.Series([0.5], dtype=np.float64),
            "loss": pd.Series([50.0], dtype=np.float64),
        }
    )
    pd.testing.assert_frame_equal(metrics.df, expected_df)


def test_reset():
    metrics = MetricsHandler(
        loss=LossMetricConfig(
            loss_fn=BCELoss(),
        ),
        mse=MSEMetricConfig(),
    )

    metrics(BATCH_1)
    metrics.aggregate()
    metrics.reset(reset_df=True)
    assert len(metrics.df) == 0
    assert len(metrics.detailed_df) == 0
    metrics._callable_metrics["loss"].get_buffer() is None
    metrics._callable_metrics["mse"].get_buffer() is None

    metrics(BATCH_2)
    metrics.aggregate()
    metrics.reset()
    assert len(metrics.df) == 1
    assert len(metrics.detailed_df) == 3
    metrics._callable_metrics["loss"].get_buffer() is None
    metrics._callable_metrics["mse"].get_buffer() is None


def test_epoch():
    metrics = MetricsHandler(
        mse=MSEMetricConfig(),
    )
    metrics(BATCH_1, epoch=0)
    metrics.aggregate(epoch=0)
    metrics.reset()
    metrics(BATCH_2, epoch=1)
    metrics.aggregate(epoch=1)

    expected_detailed_df = pd.DataFrame.from_dict(
        {
            "mse": pd.Series([0.0, 1.0, 0.0, 1.0, 1.0, 0.0], dtype=np.float32),
            "participant_id": [f"sub-{i}" for i in range(6)],
            "session_id": [f"ses-{i}" for i in range(6)],
            "epoch": [0, 0, 0, 1, 1, 1],
        }
    )
    expected_df = pd.DataFrame.from_dict(
        {
            "mse": pd.Series([0.333333, 0.666666], dtype=np.float64),
            "epoch": [0, 1],
        }
    )
    pd.testing.assert_frame_equal(metrics.detailed_df, expected_detailed_df)
    pd.testing.assert_frame_equal(metrics.df, expected_df)

    metrics.reset(reset_df=True)
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
    metrics(BATCH_1)
    metrics.add_metrics(
        loss=LossMetricConfig(loss_fn=BCELoss()), my_metric=CustomMetric()
    )
    metrics(BATCH_2)
    expected_df = pd.DataFrame.from_dict(
        {
            "loss": pd.Series(
                [np.nan, np.nan, np.nan, 100.0, 100.0, 0.0], dtype=np.float32
            ),
            "mse": pd.Series([0.0, 1.0, 0.0, 1.0, 1.0, 0.0], dtype=np.float32),
            "my_metric": pd.Series(
                [np.nan, np.nan, np.nan, 0.0, 0.0, 1.0], dtype=np.float32
            ),
            "participant_id": [f"sub-{i}" for i in range(6)],
            "session_id": [f"ses-{i}" for i in range(6)],
        }
    )
    pd.testing.assert_frame_equal(metrics.detailed_df, expected_df)


def test_checks():
    with pytest.raises(ValidationError):
        MetricsHandler(
            mse=lambda x: x,
        )
    with pytest.raises(
        ValueError, match="Loss must be passed as a LossMetricConfig. Got function"
    ):
        MetricsHandler(
            loss=lambda x: x,
        )

    config = MetricsHandler(
        mse=MSEMetricConfig(),
    )
    with pytest.raises(ValueError, match="A metric named 'mse' already exists!"):
        config.add_metrics(mse=MSEMetricConfig())
    with pytest.raises(ValidationError):
        config.add_metrics(my_metric=lambda x: x)

    with pytest.raises(
        ValueError,
        match="Loss must be passed as a LossMetricConfig. Got MSEMetricConfig",
    ):
        config.add_metrics(loss=MSEMetricConfig())
    config.add_metrics(loss=LossMetricConfig(loss_fn=BCELoss()))
    with pytest.raises(ValueError, match="You already passed a loss!"):
        config.add_metrics(loss=LossMetricConfig(loss_fn=BCELoss()))


def test_save_df(tmp_path):
    metrics = MetricsHandler(
        loss=LossMetricConfig(loss_fn=BCELoss()),
        mse=MSEMetricConfig(),
    )
    metrics(BATCH_1)
    metrics.aggregate()
    metrics.save(tmp_path / "df.tsv")
    df = pd.read_csv(tmp_path / "df.tsv", sep="\t")
    expected_df = pd.DataFrame.from_dict(
        {
            "mse": pd.Series([0.3333333]),
            "loss": pd.Series([33.33333]),
        }
    )
    pd.testing.assert_frame_equal(df, expected_df)

    metrics.save(tmp_path / "df.tsv", details_path=tmp_path / "detailed_df.tsv")
    df = pd.read_csv(tmp_path / "detailed_df.tsv", sep="\t")
    expected_df = pd.DataFrame.from_dict(
        {
            "mse": pd.Series([0.0, 1.0, 0.0]),
            "loss": pd.Series([0.0, 100.0, 0.0]),
            "participant_id": [f"sub-{i}" for i in range(3)],
            "session_id": [f"ses-{i}" for i in range(3)],
        }
    )
    pd.testing.assert_frame_equal(df, expected_df)


def test_read_write_json(tmp_path):
    metrics = MetricsHandler(
        loss=LossMetricConfig(loss_fn=BCELoss()),
        mse=MSEMetricConfig(),
    )
    metrics.add_metrics(my_metric=CustomMetric())
    metrics.write_json(tmp_path / "metrics.json")

    excepted_dict = {
        "metrics": {
            "mse": {
                "name": "MSEMetric",
                "get_not_nans": False,
                "pred_key": "output",
                "label_key": "label",
                "postprocessing": [],
                "reduction": "mean",
            },
            "my_metric": "Custom metric passed by the user: 'CustomMetric'",
        }
    }
    with open(tmp_path / "metrics.json", "r") as f:
        d = json.load(f)
    assert d == excepted_dict

    with pytest.raises(ValueError, match="Custom metric found for 'my_metric' in*"):
        MetricsHandler.from_json(json_path=tmp_path / "metrics.json")

    metrics = MetricsHandler.from_json(
        json_path=tmp_path / "metrics.json",
        my_metric=CustomMetric(),
        loss=LossMetricConfig(loss_fn=BCELoss()),
    )
    assert isinstance(metrics.metrics["mse"], MSEMetricConfig)
    assert isinstance(metrics.metrics["my_metric"], CustomMetric)
    assert isinstance(metrics.metrics["loss"], LossMetricConfig)
