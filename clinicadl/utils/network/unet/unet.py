import torch
from torch import nn

from clinicadl.utils.network.network import Network
from clinicadl.utils.network.vae.vae_layers import Flatten, Unflatten3D


class Unet_recon(Network):
    def __init__(self, gpu=True):
        super(Unet_recon, self).__init__(gpu=gpu)

        self.down1 = UNetDown(1, 32).to(self.device)
        self.down2 = UNetDown(32, 64).to(self.device)
        self.down3 = UNetDown(64, 128).to(self.device)
        self.down4 = UNetDown(128, 256).to(self.device)
        self.down5 = UNetDown(256, 512).to(self.device)

        self.flatten = Flatten().to(self.device)
        self.densedown = nn.Linear(76800, 256).to(self.device)

        self.denseup = nn.Linear(256, 76800).to(self.device)
        self.unflatten = Unflatten3D(512, 5, 6, 5).to(self.device)

        self.up1 = UNetUp(512, 256, [10, 13, 11]).to(self.device)
        self.up2 = UNetUp(512, 128, [21, 26, 22]).to(self.device)
        self.up3 = UNetUp(256, 64, [42, 52, 44]).to(self.device)
        self.up4 = UNetUp(128, 32, [84, 104, 89]).to(self.device)

        self.final = FinalLayer(64, 1).to(self.device)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        df = self.flatten(d5)
        z = self.densedown(df)

        uf = self.denseup(z)
        u0 = self.unflatten(uf)
        u1 = self.up1(u0)
        u2 = self.up2(u1, d4)
        u3 = self.up3(u2, d3)
        u4 = self.up4(u3, d2)

        return self.final(u4, d1)

    def predict(self, x):
        img = x["data"].to(self.device)
        recon_x = self.forward(img)
        return {"recon_x": recon_x}

    def compute_outputs_and_loss(self, input_dict, criterion, use_labels=True):

        images = input_dict["data"].to(self.device)
        recon_images = self.forward(images)

        loss = criterion(recon_images, images)

        return recon_images, {"loss": loss}

    @property
    def layers(self):
        return torch.nn.Sequential(
            self.down1,
            self.down2,
            self.down3,
            self.down4,
            self.down5,
            self.flatten,
            self.densedown,
            self.denseup,
            self.unflatten,
            self.up1,
            self.up2,
            self.up3,
            self.up4,
            self.final
        )

class UNetDown(nn.Module):
    """Descending block of the U-Net.

    Args:
        in_size: (int) number of channels in the input image.
        out_size : (int) number of channels in the output image.

    """

    def __init__(self, in_size, out_size):
        super(UNetDown, self).__init__()
        self.model = nn.Sequential(
            nn.Conv3d(in_size, out_size, kernel_size=4, stride=2, padding=1, bias=False,),
            nn.BatchNorm3d(out_size),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    """Ascending block of the U-Net.

    Args:
        in_channels: (int) number of channels in the input image.
        out_channels : (int) number of channels in the output image.

    """

    def __init__(self, in_channels, out_channels, up_size):
        super(UNetUp, self).__init__()
        self.model = nn.Sequential(
            nn.Upsample(size=up_size, mode='nearest'),
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False,),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip_input=None):
        if skip_input is not None:
            x = torch.cat((x, skip_input), 1)  # add the skip connection
        x = self.model(x)
        return x


class FinalLayer(nn.Module):
    """Final block of the U-Net.

    Args:
        in_size: (int) number of channels in the input image.
        out_size : (int) number of channels in the output image.

    """

    def __init__(self, in_size, out_size):
        super(FinalLayer, self).__init__()
        self.model = nn.Sequential(
            nn.Upsample(size=[169, 208, 179]),
            nn.Conv3d(
                in_size,
                out_size,
                3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.Sigmoid(),
        )

    def forward(self, x, skip_input=None):
        if skip_input is not None:
            x = torch.cat((x, skip_input), 1)  # add the skip connection
        x = self.model(x)
        return x
