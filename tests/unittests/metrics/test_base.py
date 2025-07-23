import torch

from clinicadl.metrics import Metric

Y_1 = torch.tensor([[0, 1], [1, 0], [1, 0], [0, 1]])
Y_1_PRED = torch.tensor([[0, 1], [1, 0], [0, 1], [1, 0]])

Y_2 = torch.tensor([[0, 1], [1, 0], [1, 0], [0, 1]])
Y_2_PRED = torch.tensor([[1, 0], [0, 1], [0, 1], [1, 0]])

Y_3 = torch.tensor([[0, 1], [1, 0], [1, 0], [0, 1]])
Y_3_PRED = torch.tensor([[0, 1], [0, 1], [0, 1], [1, 0]])


class TestMetric(Metric):
    def _accumulate(self, y_pred, y=None):
        return (y == y_pred).all(1)

    def _aggregate(self, data):
        return data.float().mean()


def test_metric():
    metric = TestMetric()
    assert (metric(Y_1_PRED, Y_1) == torch.tensor([1.0, 1.0, 0.0, 0.0])).all()
    assert (metric(Y_2_PRED, Y_2) == torch.tensor([0.0, 0.0, 0.0, 0.0])).all()
    assert (metric(Y_3_PRED, Y_3) == torch.tensor([1.0, 0.0, 0.0, 0.0])).all()
    assert metric.aggregate() == 0.25
