import torch
import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class SE_Blocks(nn.Module):
    def __init__(self, num_channels, ratio_channel):
        super(SE_Blocks, self).__init__()
        self.num_channels = num_channels
        self.avg_pooling_3D = nn.AdaptiveAvgPool3d(1)
        num_channels_reduced = num_channels // ratio_channel
        self.fc1 = nn.Linear(num_channels, num_channels_reduced)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels)
        self.act1 = nn.ReLU()
        self.act2 = nn.Sigmoid()

    def forward(self, input_tensor):
        """
        Parameters
        ----------
        input_tensor: pt tensor
            X, shape = (batch_size, num_channels, D, H, W)

        Returns
        -------
        output_tensor: pt tensor
        """
        batch_size, num_channels, D, H, W = input_tensor.size()
        # Average along each channel
        squeeze_tensor = self.avg_pooling_3D(input_tensor)

        # channel excitation
        fc_out_1 = self.act1(self.fc1(squeeze_tensor.view(batch_size, num_channels)))
        fc_out_2 = self.act2(self.fc2(fc_out_1))

        output_tensor = torch.mul(
            input_tensor, fc_out_2.view(batch_size, num_channels, 1, 1, 1)
        )

        return output_tensor


class ResBlock_SE(nn.Module):
    def __init__(self, block_number, input_size, num_channels, ration_channel=8):
        super(ResBlock_SE, self).__init__()

        layer_in = input_size if input_size is not None else 2 ** (block_number + 1)
        layer_out = 2 ** (block_number + 2)

        self.conv1 = nn.Conv3d(
            layer_in, layer_out, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm3d(layer_out)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv3d(
            layer_out, layer_out, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm3d(layer_out)

        self.se_block = SE_Blocks(num_channels, ration_channel)

        # shortcut
        self.shortcut = nn.Sequential(
            nn.Conv3d(
                layer_in, layer_out, kernel_size=1, stride=1, padding=0, bias=False
            )
        )

        self.act2 = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se_block(out)
        out += self.shortcut(x)
        out = self.act2(out)
        return out


class SECNNDesigner3D(nn.Module):
    def __init__(self, input_size=[1, 169, 208, 179]):
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
            Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(128 * d * h * w, 256),  # t1 image
            nn.ReLU(),
            nn.Linear(256, 2),
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
