from copy import deepcopy

import torch
import torchio as tio

from clinicadl.data.structures import DataPoint
from clinicadl.metrics import Metric

DATAPOINT = DataPoint(
    image=tio.ScalarImage(tensor=torch.randn(1, 2, 2, 2)),
    label=None,
    participant="abc",
    session="abc",
)

Y_1 = torch.tensor([[0, 1], [1, 0], [1, 0], [0, 1]])
Y_1_PRED = torch.tensor([[0, 1], [1, 0], [0, 1], [1, 0]])

Y_2 = torch.tensor([[0, 1], [1, 0], [1, 0], [0, 1]])
Y_2_PRED = torch.tensor([[1, 0], [0, 1], [0, 1], [1, 0]])

Y_3 = torch.tensor([[0, 1], [1, 0], [1, 0], [0, 1]])
Y_3_PRED = torch.tensor([[0, 1], [0, 1], [0, 1], [1, 0]])


class TestMetric(Metric):
    _optimum = "max"

    def _accumulate(self, batch):
        pred = torch.stack([datapoint["output"] for datapoint in batch])
        gt = torch.stack([datapoint["label"] for datapoint in batch])
        return (pred == gt).all(1)

    def _aggregate(self, data):
        return data.float().mean()


def test_metric():
    metric = TestMetric()

    batches = ((Y_1_PRED, Y_1), (Y_2_PRED, Y_2), (Y_3_PRED, Y_3))
    results = (
        torch.tensor([1.0, 1.0, 0.0, 0.0]),
        torch.tensor([0.0, 0.0, 0.0, 0.0]),
        torch.tensor([1.0, 0.0, 0.0, 0.0]),
    )

    for (preds, ground_truths), result in zip(batches, results):
        batch = [deepcopy(DATAPOINT) for _ in range(len(preds))]
        for datapoint, pred, gt in zip(batch, preds, ground_truths):
            datapoint["label"] = gt
            datapoint["output"] = pred

        assert (metric(batch) == result).all()

    print(metric.get_buffer())

    assert metric.aggregate() == 0.25

    assert metric.optimum == "max"
