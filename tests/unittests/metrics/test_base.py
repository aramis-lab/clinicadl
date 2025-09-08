import pytest
import torch
import torchio as tio

from clinicadl.data.dataloader import Batch
from clinicadl.data.structures import DataPoint
from clinicadl.metrics import Metric
from tests.utils import ddp_test, ddp_wrapper

DATAPOINT = DataPoint(
    image=tio.ScalarImage(tensor=torch.randn(1, 2, 2, 2)),
    label=None,
    participant="abc",
    session="abc",
)

GTS = [[0, 1], [1, 0], [1, 0], [1, 0], [0, 1], [1, 0]]
PREDS = [[0, 1], [1, 0], [0, 1], [1, 0], [1, 0], [0, 1]]

BATCH = [
    DataPoint(
        image=tio.ScalarImage(tensor=torch.randn(1, 2, 2, 2)),
        label=torch.tensor(gt),
        output=torch.tensor(pred),
        participant="abc",
        session="abc",
    )
    for pred, gt in zip(PREDS, GTS)
]


class CustomTestMetric(Metric):
    _optimum = "max"

    def _accumulate(self, batch):
        pred = torch.stack([datapoint["output"] for datapoint in batch])
        gt = torch.stack([datapoint["label"] for datapoint in batch])
        return (pred == gt).all(1)

    def _aggregate(self, data):
        return data.float().mean()


def test_metric():
    metric = CustomTestMetric()

    batches = [Batch(BATCH[:2]), Batch(BATCH[2:4]), Batch(BATCH[4:])]
    results = (
        torch.tensor([1.0, 1.0]),
        torch.tensor([0.0, 1.0]),
        torch.tensor([0.0, 0.0]),
    )
    for batch, result in zip(batches, results):
        assert (metric(batch) == result).all()

    assert metric.aggregate() == 0.5

    assert metric.optimum == "max"


@ddp_wrapper
def metric_parallelism(rank):
    batches = [Batch(BATCH[:2]), Batch(BATCH[2:4]), Batch(BATCH[4:])]
    metric = CustomTestMetric()
    if rank == 0:
        assert rank == 0
        metric(batches[0].to(rank))
        metric(batches[2].to(rank))
    elif rank == 1:
        batches[1].to(rank)
        metric(batches[1].to(rank))

    assert metric.aggregate() == 0.5


@pytest.mark.multi_gpu
def test_metric_dpp():
    ddp_test(metric_parallelism, world_size=2)
