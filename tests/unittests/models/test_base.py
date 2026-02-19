import pytest
import torch

from clinicadl.models import Model
from clinicadl.networks.nn import MLP


class MyModel(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.network = MLP(num_inputs=1, num_outputs=1, norm="batch", hidden_dims=[1])

    def forward(self, x):
        return self.network(x)

    def forward_step(self, batch):
        pass

    def backward_step(
        self,
        loss,
        grad_scaler,
    ) -> None:
        pass

    def optimization_step(
        self,
        optimizers,
        grad_scaler,
    ) -> None:
        pass

    def evaluation_step(self, batch):
        pass

    def prediction_step(self, batch):
        pass

    def build_optimizers(self):
        pass

    def get_loss_functions(self):
        pass


def test_reset():
    model = MyModel()
    model.train()
    init_param = next(iter(model.network.hidden0.linear.parameters())).clone()
    model(torch.ones(2, 1))
    assert not torch.equal(model.network.hidden0.adn.N.running_mean, torch.zeros(1))

    model.reset()

    new_param = next(iter(model.network.hidden0.linear.parameters()))
    with pytest.raises(AssertionError):
        torch.testing.assert_close(init_param, new_param)
    assert torch.equal(model.network.hidden0.adn.N.running_mean, torch.zeros(1))
