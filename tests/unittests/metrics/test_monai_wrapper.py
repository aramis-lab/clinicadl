import pytest
import torch
from monai.metrics import (
    AveragePrecisionMetric,
    ConfusionMatrixMetric,
    DiceMetric,
    MSEMetric,
)

from clinicadl.metrics.monai_wrapper import MonaiMetricWrapper


@pytest.mark.parametrize(
    "monai_metric,y_1,y_2,pred,intermediate_1,intermediate_2,final",
    [
        (
            AveragePrecisionMetric(),
            torch.tensor([1]),
            torch.tensor([0]),
            torch.tensor([0.6]),
            float("nan"),
            float("nan"),
            0.6667,
        ),
        (
            ConfusionMatrixMetric(metric_name="accuracy"),
            torch.tensor([[1]]),
            torch.tensor([[0]]),
            torch.tensor([[1]]),
            1.0000,
            0.0000,
            0.6667,
        ),
        (
            MSEMetric(),
            torch.tensor([1, 0]).repeat(1, 1, 2, 2, 2),
            torch.tensor([0, 1]).repeat(1, 1, 2, 2, 2),
            torch.tensor([1, 0]).repeat(1, 1, 2, 2, 2),
            0.0000,
            1.0000,
            0.3333,
        ),
        (
            DiceMetric(),
            torch.tensor([1, 0]).repeat(1, 1, 2, 2, 2),
            torch.tensor([0, 1]).repeat(1, 1, 2, 2, 2),
            torch.tensor([1, 0]).repeat(1, 1, 2, 2, 2),
            1.0000,
            0.0000,
            0.6667,
        ),
    ],
)
def test_monai_metric_wrapper(
    monai_metric, y_1, y_2, pred, intermediate_1, intermediate_2, final
):
    metric = MonaiMetricWrapper(monai_metric)
    torch.testing.assert_close(
        metric(pred, y_1), torch.tensor([intermediate_1]), equal_nan=True
    )
    torch.testing.assert_close(
        metric(
            torch.cat([pred, pred]),
            torch.cat([y_1, y_2]),
        ),
        torch.tensor([intermediate_1, intermediate_2]),
        equal_nan=True,
    )
    torch.testing.assert_close(metric.aggregate(), final, rtol=1e-4, atol=1e-4)
