# Adapted from https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
# and https://github.com/julianstastny/VAE-ResNet18-PyTorch/tree/master

from typing import List, Tuple

import torch
import torch.nn.functional as F
from clinicadl.utils.network.pythae.pythae_utils import BasePythae
from clinicadl.utils.network.vae.vae_layers import Flatten, Unflatten3D
from pythae.models.base.base_utils import ModelOutput
from pythae.models.nn import BaseEncoder, BaseDecoder
from torch import nn


class ResNet18_VAE(BasePythae):
    def __init__(
        self,
        input_size,
        first_layer_channels,
        n_block_encoder,
        feature_size,
        latent_space_size,
        n_block_decoder,
        last_layer_channels,
        last_layer_conv,
        n_layer_per_block_encoder,
        n_layer_per_block_decoder,
        block_type,
        gpu=False,
    ):
        from pythae.models import VAE, VAEConfig

        _, _ = super(ResNet18_VAE, self).__init__(
            input_size=input_size,
            first_layer_channels=first_layer_channels,
            n_block_encoder=n_block_encoder,
            feature_size=feature_size,
            latent_space_size=latent_space_size,
            n_block_decoder=n_block_decoder,
            last_layer_channels=last_layer_channels,
            last_layer_conv=last_layer_conv,
            n_layer_per_block_encoder=n_layer_per_block_encoder,
            n_layer_per_block_decoder=n_layer_per_block_decoder,
            block_type=block_type,
            gpu=gpu,
        )

        encoder = ResNet18Enc(
            latent_space_size=latent_space_size, input_size=input_size
        )
        decoder = ResNet18Dec(
            latent_space_size=latent_space_size, output_size=input_size
        )

        # encoder = Encoder_VAE(encoder_layers, mu_layer, logvar_layer)
        # decoder = Decoder(decoder_layers)

        model_config = VAEConfig(
            input_dim=self.input_size,
            latent_dim=self.latent_space_size,
            uses_default_encoder=False,
            uses_default_decoder=False,
        )

        self.model = VAE(
            model_config=model_config,
            encoder=encoder,
            decoder=decoder,
        )

    def get_trainer_config(self, output_dir, num_epochs, learning_rate, batch_size):
        from pythae.trainers import BaseTrainerConfig

        return BaseTrainerConfig(
            output_dir=output_dir,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            # amp=True,
        )


def conv3x3x3(input_channels: int, output_channels: int, stride: int = 1) -> nn.Conv3d:
    """3x3x3 convolution with padding"""
    return nn.Conv3d(
        input_channels,
        output_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


def conv1x1x1(input_channels: int, output_channels: int, stride: int = 1) -> nn.Conv3d:
    """1x1x1 convolution"""
    return nn.Conv3d(
        input_channels, output_channels, kernel_size=1, stride=stride, bias=False
    )


class BasicBlockEnc(nn.Module):
    def __init__(self, input_channels: int, stride: int = 1):
        super(BasicBlockEnc, self).__init__()

        output_channels = input_channels * stride

        self.conv1 = conv3x3x3(
            input_channels,
            output_channels,
            stride=stride,
        )
        self.bn1 = nn.BatchNorm3d(output_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3x3(output_channels, output_channels, stride=1)
        self.bn2 = nn.BatchNorm3d(output_channels)

        if stride == 1:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                conv1x1x1(input_channels, output_channels, stride=stride),
                nn.BatchNorm3d(output_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # print("After (layer x (y) 1st conv)", out.shape)

        out = self.conv2(out)
        out = self.bn2(out)
        # print("After (layer x (y) 2nd conv)", out.shape)

        out += self.shortcut(x)
        out = self.relu(out)
        # print("After (layer x (y) shortcut)", out.shape)

        return out


class BasicBlockDec(nn.Module):
    def __init__(self, input_channels: int, up_size: List[int], stride: int = 1):
        super(BasicBlockDec, self).__init__()

        output_channels = input_channels // stride

        self.conv2 = conv3x3x3(
            input_channels,
            input_channels,
            stride=1,
        )
        self.bn2 = nn.BatchNorm3d(input_channels)

        self.bn1 = nn.BatchNorm3d(output_channels)

        if stride == 1:
            self.conv1 = conv3x3x3(
                input_channels,
                output_channels,
                stride=1,
            )
            self.shortcut = nn.Sequential()
        else:
            self.conv1 = nn.Sequential(
                nn.Upsample(size=up_size, mode="nearest"),
                conv3x3x3(
                    input_channels,
                    output_channels,
                    stride=1,
                ),
            )
            self.shortcut = nn.Sequential(
                nn.Upsample(size=up_size, mode="nearest"),
                conv3x3x3(
                    input_channels,
                    output_channels,
                    stride=1,
                ),
                nn.BatchNorm3d(output_channels),
            )

    def forward(self, x):
        out = F.relu(self.bn2(self.conv2(x)))
        # print("After layer (x (y) 1st conv)", out.shape)
        out = self.bn1(self.conv1(out))
        # print("After layer (x (y) 2nd conv)", out.shape)
        out += self.shortcut(x)
        # print("After layer (x (y) shortcut)", out.shape)
        out = F.relu(out)
        return out


class ResNet18Enc(BaseEncoder):
    def __init__(
        self,
        input_size: Tuple[int],
        latent_space_size: int,
        num_blocks: List[int] = [2, 2, 2, 2],
    ):
        super(ResNet18Enc, self).__init__()

        input_c = input_size[0]
        input_d = input_size[1]
        input_h = input_size[2]
        input_w = input_size[3]

        self.input_channels = 64
        self.latent_space_size = latent_space_size

        self.layer1 = nn.Sequential(
            nn.Conv3d(
                input_c,
                self.input_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            ),
            nn.BatchNorm3d(self.input_channels),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.layer2 = self._make_layer(64, num_blocks[0], stride=1)
        self.layer3 = self._make_layer(128, num_blocks[1], stride=2)
        self.layer4 = self._make_layer(256, num_blocks[2], stride=2)
        self.layer5 = self._make_layer(512, num_blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.flatten = Flatten()

        # n_pix_encoder = 512 * 6 * 7 * 6
        n_pix_encoder = 512
        feature_size = n_pix_encoder

        self.mu_layer = nn.Linear(feature_size, self.latent_space_size)
        self.logvar_layer = nn.Linear(feature_size, self.latent_space_size)

    def _make_layer(self, output_channels: int, num_blocks: int, stride: int):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers += [BasicBlockEnc(self.input_channels, stride)]
            self.input_channels = output_channels
        return nn.Sequential(*layers)

    def forward(self, x: torch.torch.Tensor) -> ModelOutput:
        # print("ResNet18Enc")

        x = self.layer1(x)
        # print("After (layer 1)", x.shape)

        x = self.maxpool(x)
        # print("After (maxpool)", x.shape)

        x = self.layer2(x)
        # print("After (layer 2)", x.shape)

        x = self.layer3(x)
        # print("After (layer 3)", x.shape)

        x = self.layer4(x)
        # print("After (layer 4)", x.shape)

        x = self.layer5(x)
        # print("After (layer 5)", x.shape)

        x = self.avgpool(x)
        # print("After (avgpool)", x.shape)
        x = self.flatten(x)
        # print("After (flatten)", x.shape)

        mu = self.mu_layer(x)
        # print("After mu", mu.shape)
        logvar = self.logvar_layer(x)
        # print("After logvar", logvar.shape)

        output = ModelOutput(
            embedding=mu,
            log_covariance=logvar,
        )
        return output


class ResNet18Dec(BaseDecoder):
    def __init__(
        self,
        output_size: Tuple[int],
        latent_space_size: int,
        num_blocks: List[int] = [2, 2, 2, 2],
    ):
        super(ResNet18Dec, self).__init__()

        self.input_channels = 512

        # n_pix_decoder = 512 * 6 * 7 * 6
        n_pix_decoder = 512
        feature_size = n_pix_decoder

        self.linear = nn.Sequential(
            nn.Linear(latent_space_size, feature_size),
            nn.ReLU(inplace=True),
        )

        self.unflatten = Unflatten3D(512, 1, 1, 1)

        self.upsample = nn.Upsample(size=[6, 7, 6], mode="nearest")

        self.layer5 = self._make_layer(
            256, num_blocks[3], stride=2, up_size=[11, 13, 12]
        )
        self.layer4 = self._make_layer(
            128, num_blocks[2], stride=2, up_size=[22, 26, 23]
        )
        self.layer3 = self._make_layer(
            64, num_blocks[1], stride=2, up_size=[43, 52, 45]
        )
        self.layer2 = self._make_layer(
            64, num_blocks[0], stride=1, up_size=[85, 104, 90]
        )

        self.layer1 = nn.Sequential(
            nn.Upsample(size=[169, 208, 179], mode="nearest"),
            conv3x3x3(64, output_size[0], 1),
            nn.Sigmoid(),
        )

    def _make_layer(
        self,
        output_channels: int,
        num_blocks: int,
        stride: int,
        up_size: List[int],
    ):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in reversed(strides):
            layers += [
                BasicBlockDec(self.input_channels, up_size=up_size, stride=stride)
            ]
        self.input_channels = output_channels
        return nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> ModelOutput:
        # print("ResNet18Dec")

        # print("Latent vector", z.shape)
        x = self.linear(z)
        # print("After (linear)", x.shape)

        x = self.unflatten(x)
        # print("After (unflatten)", x.shape)
        x = self.upsample(x)
        # print("After (upsample)", x.shape)

        out = self.layer5(x)
        print("After (layer 5)", out.shape)

        out = self.layer4(out)
        # print("After (layer 4)", out.shape)

        out = self.layer3(out)
        # print("After (layer 3)", out.shape)

        out = self.layer2(out)
        # print("After (layer 2)", out.shape)

        out = self.layer1(out)
        # print("After (layer 1)", out.shape)

        output = ModelOutput(reconstruction=out)

        return output

