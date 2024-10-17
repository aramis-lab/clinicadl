import numpy as np
import torch
import torch.utils.model_zoo as model_zoo
from torch import nn
from torchvision.models.resnet import BasicBlock

from clinicadl.nn.layers.factory import (
    get_conv_layer,
    get_norm_layer,
    get_pool_layer,
)
from clinicadl.utils.enum import BaseEnum

from .factory import ResNetDesigner, ResNetDesigner3D, SECNNDesigner3D
from .factory.resnet import model_urls


class CNN2d(str, BaseEnum):
    """CNNs compatible with 2D inputs."""

    CONV5_FC3 = "Conv5_FC3"
    CONV4_FC3 = "Conv4_FC3"
    STRIDE_CONV5_FC3 = "Stride_Conv5_FC3"
    RESNET = "resnet18"


class CNN3d(str, BaseEnum):
    """CNNs compatible with 3D inputs."""

    CONV5_FC3 = "Conv5_FC3"
    CONV4_FC3 = "Conv4_FC3"
    STRIDE_CONV5_FC3 = "Stride_Conv5_FC3"
    RESNET3D = "ResNet3D"
    SECNN = "SqueezeExcitationCNN"


class ImplementedCNN(str, BaseEnum):
    """Implemented CNNs in ClinicaDL."""

    CONV5_FC3 = "Conv5_FC3"
    CONV4_FC3 = "Conv4_FC3"
    STRIDE_CONV5_FC3 = "Stride_Conv5_FC3"
    RESNET = "resnet18"
    RESNET3D = "ResNet3D"
    SECNN = "SqueezeExcitationCNN"

    @classmethod
    def _missing_(cls, value):
        raise ValueError(
            f"{value} is not implemented. Implemented CNNs are: "
            + ", ".join([repr(m.value) for m in cls])
        )


# Networks #
class CNN(nn.Module):
    """Base class for CNN."""

    def __init__(self, convolution_layers: nn.Module, fc_layers: nn.Module) -> None:
        super().__init__()
        self.convolutions = convolution_layers
        self.fc = fc_layers

    def forward(self, x):
        inter = self.convolutions(x)
        return self.fc(inter)


class Conv5_FC3(CNN):
    """A Convolutional Neural Network with 5 convolution and 3 fully-connected layers."""

    def __init__(self, input_size, output_size, dropout):
        dim = len(input_size) - 1
        in_channels = input_size[0]

        conv = get_conv_layer(dim)
        pool = get_pool_layer("PadMaxPool", dim=dim)
        norm = get_norm_layer("BatchNorm", dim=dim)

        convolutions = nn.Sequential(
            conv(in_channels, 8, 3, padding=1),
            norm(8),
            nn.ReLU(),
            pool(2, 2),
            conv(8, 16, 3, padding=1),
            norm(16),
            nn.ReLU(),
            pool(2, 2),
            conv(16, 32, 3, padding=1),
            norm(32),
            nn.ReLU(),
            pool(2, 2),
            conv(32, 64, 3, padding=1),
            norm(64),
            nn.ReLU(),
            pool(2, 2),
            conv(64, 128, 3, padding=1),
            norm(128),
            nn.ReLU(),
            pool(2, 2),
        )

        input_tensor = torch.zeros(input_size).unsqueeze(0)
        output_shape = convolutions(input_tensor).shape

        fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(np.prod(list(output_shape)).item(), 1300),
            nn.ReLU(),
            nn.Linear(1300, 50),
            nn.ReLU(),
            nn.Linear(50, output_size),
        )
        super().__init__(convolutions, fc)


class Conv4_FC3(CNN):
    """A Convolutional Neural Network with 4 convolution and 3 fully-connected layers."""

    def __init__(self, input_size, output_size, dropout):
        dim = len(input_size) - 1
        in_channels = input_size[0]

        conv = get_conv_layer(dim)
        pool = get_pool_layer("PadMaxPool", dim=dim)
        norm = get_norm_layer("BatchNorm", dim=dim)

        convolutions = nn.Sequential(
            conv(in_channels, 8, 3, padding=1),
            norm(8),
            nn.ReLU(),
            pool(2, 2),
            conv(8, 16, 3, padding=1),
            norm(16),
            nn.ReLU(),
            pool(2, 2),
            conv(16, 32, 3, padding=1),
            norm(32),
            nn.ReLU(),
            pool(2, 2),
            conv(32, 64, 3, padding=1),
            norm(64),
            nn.ReLU(),
            pool(2, 2),
            conv(64, 128, 3, padding=1),
            norm(128),
            nn.ReLU(),
            pool(2, 2),
        )

        input_tensor = torch.zeros(input_size).unsqueeze(0)
        output_shape = convolutions(input_tensor).shape

        fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(np.prod(list(output_shape)).item(), 50),
            nn.ReLU(),
            nn.Linear(50, 40),
            nn.ReLU(),
            nn.Linear(40, output_size),
        )
        super().__init__(convolutions, fc)


class Stride_Conv5_FC3(CNN):
    """A Convolutional Neural Network with 5 convolution and 3 fully-connected layers and a stride of 2 for each convolutional layer."""

    def __init__(self, input_size, output_size, dropout):
        dim = len(input_size) - 1
        in_channels = input_size[0]

        conv = get_conv_layer(dim)
        norm = get_norm_layer("BatchNorm", dim=dim)

        convolutions = nn.Sequential(
            conv(in_channels, 8, 3, padding=1, stride=2),
            norm(8),
            nn.ReLU(),
            conv(8, 16, 3, padding=1, stride=2),
            norm(16),
            nn.ReLU(),
            conv(16, 32, 3, padding=1, stride=2),
            norm(32),
            nn.ReLU(),
            conv(32, 64, 3, padding=1, stride=2),
            norm(64),
            nn.ReLU(),
            conv(64, 128, 3, padding=1, stride=2),
            norm(128),
            nn.ReLU(),
        )

        input_tensor = torch.zeros(input_size).unsqueeze(0)
        output_shape = convolutions(input_tensor).shape

        fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(np.prod(list(output_shape)).item(), 1300),
            nn.ReLU(),
            nn.Linear(1300, 50),
            nn.ReLU(),
            nn.Linear(50, output_size),
        )
        super().__init__(convolutions, fc)


class resnet18(CNN):
    """
    ResNet-18 is a neural network that is 18 layers deep based on residual block.
    It uses skip connections or shortcuts to jump over some layers.
    It is an image classification pre-trained model.
    The model input has 3 channels in RGB order.

    Reference: Kaiming He et al., Deep Residual Learning for Image Recognition.
    https://arxiv.org/abs/1512.03385?context=cs
    """

    def __init__(self, input_size, output_size, dropout):
        model = ResNetDesigner(input_size, BasicBlock, [2, 2, 2, 2])
        model.load_state_dict(model_zoo.load_url(model_urls["resnet18"]))

        convolutions = nn.Sequential(
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

        # add a fc layer on top of the transfer_learning model and a softmax classifier
        fc = nn.Sequential(nn.Flatten(), model.fc)
        fc.add_module("drop_out", nn.Dropout(p=dropout))
        fc.add_module("fc_out", nn.Linear(1000, output_size))

        super().__init__(convolutions, fc)


class ResNet3D(CNN):
    """
    ResNet3D is a 3D neural network composed of 5 residual blocks. Each residual block
    is compose of 3D convolutions followed by a batch normalization and an activation function.
    It uses skip connections or shortcuts to jump over some layers. It's a 3D version of the
    original implementation of Kaiming He et al.

    Reference: Kaiming He et al., Deep Residual Learning for Image Recognition.
    https://arxiv.org/abs/1512.03385?context=cs
    """

    def __init__(self, input_size, output_size, dropout):
        model = ResNetDesigner3D(input_size, output_size, dropout)
        convolutions = nn.Sequential(
            model.layer0, model.layer1, model.layer2, model.layer3, model.layer4
        )
        fc_layers = model.fc
        super().__init__(convolutions, fc_layers)


class SqueezeExcitationCNN(CNN):
    """
    SE-CNN is a combination of a ResNet-101 with Squeeze and Excitation blocks which was successfully
    tested on brain tumour classification by Ghosal et al. 2019. SE blocks are composed of a squeeze
    and an excitation step. The squeeze operation is obtained through an average pooling layer and
    provides a global understanding of each channel.

    The excitation part consists of a two-layer feed-forward network that outputs a vector of n values
    corresponding to the weights of each channel of the feature maps.

    Reference: Ghosal et al. Brain Tumor Classification Using ResNet-101 Based Squeeze and Excitation Deep Neural Network
    https://ieeexplore.ieee.org/document/8882973

    """

    def __init__(self, input_size, output_size, dropout):
        model = SECNNDesigner3D(input_size, output_size, dropout)
        convolutions = nn.Sequential(
            model.layer0, model.layer1, model.layer2, model.layer3, model.layer4
        )
        fc_layers = model.fc
        super().__init__(convolutions, fc_layers)
