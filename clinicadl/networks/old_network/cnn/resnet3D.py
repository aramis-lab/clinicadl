import torch
import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class ResBlock(nn.Module):
    def __init__(self, block_number, input_size):
        super(ResBlock, self).__init__()

        layer_in = input_size if input_size is not None else 2 ** (block_number + 1)
        layer_out = 2 ** (block_number + 2)

        self.conv1 = nn.Conv3d(
            layer_in, layer_out, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm3d(layer_out)
        self.act1 = nn.ELU()

        self.conv2 = nn.Conv3d(
            layer_out, layer_out, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm3d(layer_out)

        # shortcut
        self.shortcut = nn.Sequential(
            nn.Conv3d(
                layer_in, layer_out, kernel_size=1, stride=1, padding=0, bias=False
            )
        )

        self.act2 = nn.ELU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.act2(out)
        return out


class ResNetDesigner3D(nn.Module):
    def __init__(self, input_size=[1, 169, 208, 179]):
        super(ResNetDesigner3D, self).__init__()

        assert (
            len(input_size) == 4
        ), "Input must be in 3D with the corresponding number of channels."

        self.layer0 = self._make_block(1, input_size[0])
        self.layer1 = self._make_block(2)
        self.layer2 = self._make_block(3)
        self.layer3 = self._make_block(4)
        self.layer4 = self._make_block(5)

        input_tensor = torch.zeros(input_size).unsqueeze(0)
        out = self.layer0(input_tensor)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        d, h, w = self._maxpool_output_size(input_size[1::], nb_layers=5)
        self.fc = nn.Sequential(
            Flatten(),
            nn.Linear(128 * d * h * w, 256),  # t1 image
            nn.ELU(),
            nn.Dropout(p=0.8),
            nn.Linear(256, 2),
        )

        for layer in self.fc:
            out = layer(out)

    def _make_block(self, block_number, input_size=None):
        return nn.Sequential(
            ResBlock(block_number, input_size), nn.MaxPool3d(3, stride=2)
        )

    def _maxpool_output_size(
        self, input_size, kernel_size=(3, 3, 3), stride=(2, 2, 2), nb_layers=1
    ):
        import math

        d = math.floor((input_size[0] - kernel_size[0]) / stride[0] + 1)
        h = math.floor((input_size[1] - kernel_size[1]) / stride[1] + 1)
        w = math.floor((input_size[2] - kernel_size[2]) / stride[2] + 1)

        if nb_layers == 1:
            return d, h, w
        return self._maxpool_output_size(
            (d, h, w), kernel_size=kernel_size, stride=stride, nb_layers=nb_layers - 1
        )
