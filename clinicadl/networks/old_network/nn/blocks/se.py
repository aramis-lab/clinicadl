import torch
import torch.nn as nn


class SE_Block(nn.Module):
    def __init__(self, num_channels, ratio_channel):
        super().__init__()
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
    def __init__(self, block_number, input_size, num_channels, ratio_channel=8):
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

        self.se_block = SE_Block(layer_out, ratio_channel)

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
