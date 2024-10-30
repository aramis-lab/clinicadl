import torch.nn as nn


class CropMaxUnpool3d(nn.Module):
    def __init__(self, kernel_size, stride):
        super(CropMaxUnpool3d, self).__init__()
        self.unpool = nn.MaxUnpool3d(kernel_size, stride)

    def forward(self, f_maps, indices, padding=None):
        output = self.unpool(f_maps, indices)
        if padding is not None:
            x1 = padding[4]
            y1 = padding[2]
            z1 = padding[0]
            output = output[:, :, x1::, y1::, z1::]

        return output


class CropMaxUnpool2d(nn.Module):
    def __init__(self, kernel_size, stride):
        super(CropMaxUnpool2d, self).__init__()
        self.unpool = nn.MaxUnpool2d(kernel_size, stride)

    def forward(self, f_maps, indices, padding=None):
        output = self.unpool(f_maps, indices)
        if padding is not None:
            x1 = padding[2]
            y1 = padding[0]
            output = output[:, :, x1::, y1::]

        return output
