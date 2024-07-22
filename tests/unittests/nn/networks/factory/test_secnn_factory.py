import torch
import torch.nn as nn


def test_SECNNDesigner3D():
    from clinicadl.nn.networks.factory import SECNNDesigner3D

    input_ = torch.randn(2, 3, 100, 100, 100)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            model = SECNNDesigner3D(
                input_size=input_.shape[1:], output_size=2, dropout=0.5
            )
            self.convolutions = nn.Sequential(
                model.layer0,
                model.layer1,
                model.layer2,
                model.layer3,
                model.layer4,
            )
            self.fc = model.fc

        def forward(self, x):
            return self.fc(self.convolutions(x))

    model = Model()

    assert model(input_).shape == torch.Size([2, 2])
