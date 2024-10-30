import torch.nn as nn


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
