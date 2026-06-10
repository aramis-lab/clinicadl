import pytest
import torch

from clinicadl.models import Model
from clinicadl.networks.nn import CNN


class MyModel(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.network = CNN(
            in_shape=(1, 10, 10, 10),
            num_outputs=1,
            conv_args={"channels": [1, 1, 1], "norm": "batch"},
        )
        for x in self.network.convolutions.layer0.parameters():
            x.requires_grad = False

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
    ref_param_0 = next(
        iter(model.network.convolutions.layer0.conv.parameters())
    ).clone()
    ref_param_1 = next(
        iter(model.network.convolutions.layer1.conv.parameters())
    ).clone()
    out = model(torch.ones(1, 1, 10, 10, 10))
    out.mean().backward()
    assert not torch.equal(
        model.network.convolutions.layer0.adn.N.running_mean, torch.zeros(1)
    )
    assert not torch.equal(
        model.network.convolutions.layer1.adn.N.running_mean, torch.zeros(1)
    )
    assert (
        next(iter(model.network.convolutions.layer1.conv.parameters())).grad
    ) is not None

    model.reset()

    assert (
        next(iter(model.network.convolutions.layer1.conv.parameters())).grad
    ) is None
    new_param_0 = next(iter(model.network.convolutions.layer0.conv.parameters()))
    new_param_1 = next(iter(model.network.convolutions.layer1.conv.parameters()))
    torch.testing.assert_close(ref_param_0, new_param_0)
    with pytest.raises(AssertionError):
        torch.testing.assert_close(ref_param_1, new_param_1)
    assert not torch.equal(
        model.network.convolutions.layer0.adn.N.running_mean, torch.zeros(1)
    )
    assert torch.equal(
        model.network.convolutions.layer1.adn.N.running_mean, torch.zeros(1)
    )
