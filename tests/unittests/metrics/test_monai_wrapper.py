import re
from copy import deepcopy

import pytest
import torch
import torchio as tio
from monai.metrics import (
    AveragePrecisionMetric,
    ConfusionMatrixMetric,
    DiceMetric,
    GeneralizedDiceScore,
    HausdorffDistanceMetric,
    MAEMetric,
    MeanIoU,
    MSEMetric,
    MultiScaleSSIMMetric,
    PSNRMetric,
    RMSEMetric,
    ROCAUCMetric,
    SSIMMetric,
    SurfaceDiceMetric,
    SurfaceDistanceMetric,
)

from clinicadl.data.structures import DataPoint
from clinicadl.metrics.monai_wrapper import MonaiMetricWrapper
from clinicadl.transforms.config import AsDiscreteConfig

DATAPOINT = DataPoint(
    image=tio.ScalarImage(tensor=torch.randn(1, 2, 2, 2)),
    label=None,
    participant="abc",
    session="abc",
)


@pytest.mark.parametrize(
    "monai_metric,y_1,y_2,pred,intermediate_1,intermediate_2,final",
    [
        (
            AveragePrecisionMetric(),
            torch.tensor(1),
            torch.tensor(0),
            torch.tensor(0.6),
            float("nan"),
            float("nan"),
            0.6667,
        ),
        (
            ROCAUCMetric(),
            torch.tensor(1),
            torch.tensor(0),
            torch.tensor(0.6),
            float("nan"),
            float("nan"),
            0.5,
        ),
        (
            ConfusionMatrixMetric(metric_name="accuracy"),
            torch.tensor([1]),
            torch.tensor([0]),
            torch.tensor([1]),
            1.0000,
            0.0000,
            0.6667,
        ),
        (
            MSEMetric(),
            torch.tensor([1, 0]).repeat(1, 2, 2, 2),
            torch.tensor([0, 1]).repeat(1, 2, 2, 2),
            torch.tensor([1, 0]).repeat(1, 2, 2, 2),
            0.0000,
            1.0000,
            0.3333,
        ),
        (
            MAEMetric(),
            torch.tensor([1, 0]).repeat(1, 2, 2, 2).float(),
            torch.tensor([0, 1]).repeat(1, 2, 2, 2).float(),
            torch.tensor([1, 0]).repeat(1, 2, 2, 2).float(),
            0.0000,
            1.0000,
            0.3333,
        ),
        (
            RMSEMetric(),
            torch.tensor([1, 0]).repeat(1, 2, 2, 2),
            torch.tensor([0, 1]).repeat(1, 2, 2, 2),
            torch.tensor([1, 0]).repeat(1, 2, 2, 2),
            0.0000,
            1.0000,
            0.3333,
        ),
        (
            PSNRMetric(max_val=1),
            torch.tensor([1, 0]).repeat(1, 2, 2, 2),
            torch.tensor([0, 1]).repeat(1, 2, 2, 2),
            torch.tensor([0.1, 0]).repeat(1, 2, 2, 2),
            3.9255,
            2.9671,
            3.6060,
        ),
        (
            DiceMetric(),
            torch.tensor([1, 0]).repeat(1, 2, 2, 2),
            torch.tensor([0, 1]).repeat(1, 2, 2, 2),
            torch.tensor([1, 0]).repeat(1, 2, 2, 2),
            1.0000,
            0.0000,
            0.6667,
        ),
        (
            GeneralizedDiceScore(),
            torch.tensor([1, 0]).repeat(1, 2, 2, 2),
            torch.tensor([0, 1]).repeat(1, 2, 2, 2),
            torch.tensor([1, 0]).repeat(1, 2, 2, 2),
            1.0000,
            0.0000,
            0.6667,
        ),
        (
            SurfaceDiceMetric(class_thresholds=[0.1]),
            torch.tensor([1, 0]).repeat(1, 2, 2, 2),
            torch.tensor([0, 1]).repeat(1, 2, 2, 2),
            torch.tensor([1, 0]).repeat(1, 2, 2, 2),
            1.0000,
            0.0000,
            0.6667,
        ),
        (
            MeanIoU(),
            torch.tensor([1, 0]).repeat(1, 2, 2, 2),
            torch.tensor([0, 1]).repeat(1, 2, 2, 2),
            torch.tensor([1, 0]).repeat(1, 2, 2, 2),
            1.0000,
            0.0000,
            0.6667,
        ),
        (
            HausdorffDistanceMetric(),
            torch.tensor([1, 0]).repeat(1, 2, 2, 2),
            torch.tensor([0, 1]).repeat(1, 2, 2, 2),
            torch.tensor([1, 0]).repeat(1, 2, 2, 2),
            0.0000,
            1.0000,
            0.3333,
        ),
        (
            SurfaceDistanceMetric(),
            torch.tensor([1, 0]).repeat(1, 2, 2, 2),
            torch.tensor([0, 1]).repeat(1, 2, 2, 2),
            torch.tensor([1, 0]).repeat(1, 2, 2, 2),
            0.0000,
            1.0000,
            0.3333,
        ),
        (
            SSIMMetric(spatial_dims=3, win_size=1),
            torch.tensor([1, 0]).repeat(1, 2, 2, 2),
            torch.tensor([0, 1]).repeat(1, 2, 2, 2),
            torch.tensor([1, 0]).repeat(1, 2, 2, 2),
            1.0000,
            0.0000,
            0.6667,
        ),
        (
            MultiScaleSSIMMetric(spatial_dims=3, kernel_size=1, weights=(0.5,)),
            torch.tensor([1, 0]).repeat(1, 2, 2, 2),
            torch.tensor([0, 1]).repeat(1, 2, 2, 2),
            torch.tensor([1, 0]).repeat(1, 2, 2, 2),
            1.0000,
            0.0100,
            0.6700,
        ),
    ],
)
def test_monai_metric_wrapper(
    monai_metric, y_1, y_2, pred, intermediate_1, intermediate_2, final
):
    datapoint = deepcopy(DATAPOINT)
    datapoint["output"] = pred

    batch_1 = [deepcopy(datapoint)]
    batch_1[0]["label"] = y_1

    batch_2 = [deepcopy(datapoint)]
    batch_2[0]["label"] = y_2

    metric = MonaiMetricWrapper(
        monai_metric, label_key="label", pred_key="output", optimum="max"
    )
    torch.testing.assert_close(
        metric(batch_1),
        torch.tensor([intermediate_1]),
        equal_nan=True,
        rtol=1e-4,
        atol=1e-4,
    )
    torch.testing.assert_close(
        metric(batch_1 + batch_2),
        torch.tensor([intermediate_1, intermediate_2]),
        equal_nan=True,
        rtol=1e-4,
        atol=1e-4,
    )
    torch.testing.assert_close(metric.aggregate(), final, rtol=1e-4, atol=1e-4)

    assert metric.optimum == "max"


def test_repr():
    metric = MonaiMetricWrapper(
        AveragePrecisionMetric(), pred_key="abc", label_key="bcd", optimum="max"
    )
    pattern = r"MonaiMetricWrapper\(metric=<monai\.metrics\.average_precision\.AveragePrecisionMetric object at .*?>, optimum='max', pred_key='abc', label_key='bcd'\)"
    assert re.fullmatch(pattern, repr(metric))


def test_postprocessing():
    metric = MonaiMetricWrapper(
        ConfusionMatrixMetric(metric_name="accuracy"),
        pred_key="output",
        label_key="label",
        optimum="max",
        postprocessing=[
            AsDiscreteConfig(include=["output"], threshold=0.5),
            AsDiscreteConfig(include=["label", "output"], to_onehot=2),
        ],
    )
    batch = [deepcopy(DATAPOINT) for _ in range(2)]
    batch[0]["label"] = 1.0
    batch[0]["output"] = 0.7
    batch[1]["label"] = 1.0
    batch[1]["output"] = 0.3
    assert (metric(batch) == torch.tensor([1, 0])).all()
    assert metric.aggregate() == 0.5
