import torch
import torch.nn as nn

from clinicadl.nn.blocks import ResBlock_SE


class SECNNDesigner3D(nn.Module):
    def __init__(self, input_size, output_size, dropout):
        super(SECNNDesigner3D, self).__init__()

        assert (
            len(input_size) == 4
        ), "input must be in 3d with the corresponding number of channels"

        self.layer0 = self._make_block(1, 8, 8, input_size[0])
        self.layer1 = self._make_block(2, 16)
        self.layer2 = self._make_block(3, 32)
        self.layer3 = self._make_block(4, 64)
        self.layer4 = self._make_block(5, 128)

        input_tensor = torch.zeros(input_size).unsqueeze(0)
        out = self.layer0(input_tensor)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        d, h, w = self._maxpool_output_size(input_size[1::], nb_layers=5)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(128 * d * h * w, 256),  # t1 image
            nn.ReLU(),
            nn.Linear(256, output_size),
        )

        for layer in self.fc:
            out = layer(out)

    def _make_block(
        self, block_number, num_channels, ration_channel=8, input_size=None
    ):
        return nn.Sequential(
            ResBlock_SE(block_number, input_size, num_channels, ration_channel),
            nn.MaxPool3d(3, stride=2),
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
