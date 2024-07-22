import torch
import torch.nn as nn


def test_ResNetDesigner():
    from torchvision.models.resnet import BasicBlock

    from clinicadl.nn.networks.factory import ResNetDesigner

    input_ = torch.randn(2, 3, 100, 100)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            model = ResNetDesigner(
                input_size=input_.shape[1:],
                block=BasicBlock,
                layers=[1, 2, 3, 4],
                num_classes=2,
            )
            self.convolutions = nn.Sequential(
                model.conv1,
                model.bn1,
                model.relu,
                model.maxpool,
                model.layer1,
                model.layer2,
                model.layer3,
                model.layer4,
                model.avgpool,
            )
            self.fc = nn.Sequential(
                nn.Flatten(),
                model.fc,
            )

        def forward(self, x):
            return self.fc(self.convolutions(x))

    model = Model()

    assert model(input_).shape == torch.Size([2, 2])


def test_ResNetDesigner3D():
    from clinicadl.nn.networks.factory import ResNetDesigner3D

    input_ = torch.randn(2, 3, 100, 100, 100)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            model = ResNetDesigner3D(
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
