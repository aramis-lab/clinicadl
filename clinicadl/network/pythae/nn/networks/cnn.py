from enum import Enum

import numpy as np
import torch
import torch.utils.model_zoo as model_zoo
from torch import nn
from torchvision.models.resnet import BasicBlock

from clinicadl.network.pythae.nn.layers.factory import ConvLayer, NormLayer, PoolLayer
from clinicadl.network.pythae.nn.utils.resnet import ResNetDesigner, model_urls

# from clinicadl.network.pythae.nn.utils.resnet3D import ResNetDesigner3D
# from clinicadl.network.pythae.nn.utils.SECNN import SECNNDesigner3D
# from clinicadl.network.sub_network import CNN, CNN_SSDA


class ImplementedCNN(str, Enum):
    Conv5_FC3 = "Conv5_FC3"
    Conv4_FC3 = "Conv4_FC3"
    Stride_Conv5_FC3 = "Stride_Conv5_FC3"
    RESNET = "resnet18"

    @classmethod
    def _missing_(cls, value):
        raise ValueError(
            f"{value} is not implemented. Implemented networks are: "
            + ", ".join([repr(m.value) for m in cls])
        )


class Conv5_FC3(nn.Module):
    """A Convolutional Neural Network with 5 convolution and 3 fully-connected layers."""

    def __init__(self, input_size, output_size, dropout):
        dim = len(input_size) - 1
        in_channels = input_size[1]

        conv = ConvLayer(dim)
        norm = PoolLayer("PadMaxPool", dim=dim)
        pool = NormLayer("BatchNorm", dim=dim)

        self.convolutions = nn.Sequential(
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
        output_shape = self.convolutions(input_tensor).shape

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(np.prod(list(output_shape)).item(), 1300),
            nn.ReLU(),
            nn.Linear(1300, 50),
            nn.ReLU(),
            nn.Linear(50, output_size),
        )

    def forward(self, x):
        x = self.convolutions(x)
        return self.fc(x)


class Conv4_FC3(nn.Module):
    """A Convolutional Neural Network with 4 convolution and 3 fully-connected layers."""

    def __init__(self, input_size, output_size, dropout):
        dim = len(input_size) - 1
        in_channels = input_size[1]

        conv = ConvLayer(dim)
        norm = PoolLayer("PadMaxPool", dim=dim)
        pool = NormLayer("BatchNorm", dim=dim)

        self.convolutions = nn.Sequential(
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
        output_shape = self.convolutions(input_tensor).shape

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(np.prod(list(output_shape)).item(), 50),
            nn.ReLU(),
            nn.Linear(50, 40),
            nn.ReLU(),
            nn.Linear(40, output_size),
        )

    def forward(self, x):
        x = self.convolutions(x)
        return self.fc(x)


class Stride_Conv5_FC3(nn.Module):
    """A Convolutional Neural Network with 5 convolution and 3 fully-connected layers and a stride of 2 for each convolutional layer."""

    def __init__(self, input_size, output_size, dropout):
        dim = len(input_size) - 1
        in_channels = input_size[1]

        conv = ConvLayer(dim)
        norm = PoolLayer("PadMaxPool", dim=dim)

        self.convolutions = nn.Sequential(
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
        output_shape = self.convolutions(input_tensor).shape

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(np.prod(list(output_shape)).item(), 1300),
            nn.ReLU(),
            nn.Linear(1300, 50),
            nn.ReLU(),
            nn.Linear(50, output_size),
        )

    def forward(self, x):
        x = self.convolutions(x)
        return self.fc(x)


class resnet18(nn.Module):
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

        # add a fc layer on top of the transfer_learning model and a softmax classifier
        self.fc = nn.Sequential(nn.Flatten(), model.fc)
        self.fc.add_module("drop_out", nn.Dropout(p=dropout))
        self.fc.add_module("fc_out", nn.Linear(1000, output_size))

    def forward(self, x):
        x = self.convolutions(x)
        return self.fc(x)


# TODO : check the following networks #

# class ResNet3D(nn.Module):
#     """
#     ResNet3D is a 3D neural network composed of 5 residual blocks. Each residual block
#     is compose of 3D convolutions followed by a batch normalization and an activation function.
#     It uses skip connections or shortcuts to jump over some layers. It's a 3D version of the
#     original implementation of Kaiming He et al.

#     Reference: Kaiming He et al., Deep Residual Learning for Image Recognition.
#     https://arxiv.org/abs/1512.03385?context=cs
#     """

#     def __init__(self, input_size, dropout, output_size=1):
#         model = ResNetDesigner3D(input_size)

#         self.convolutions = nn.Sequential(
#             model.layer0, model.layer1, model.layer2, model.layer3, model.layer4
#         )

#         self.fc = model.fc

#     def forward(self, x):
#         x = self.convolutions(x)
#         return self.fc(x)


# class SqueezeExcitationCNN(CNN):
#     """
#     SE-CNN is a combination of a ResNet-101 with Squeeze and Excitation blocks which was successfully
#     tested on brain tumour classification by Ghosal et al. 2019. SE blocks are composed of a squeeze
#     and an excitation step. The squeeze operation is obtained through an average pooling layer and
#     provides a global understanding of each channel.

#     The excitation part consists of a two-layer feed-forward network that outputs a vector of n values
#     corresponding to the weights of each channel of the feature maps.

#     Reference: Ghosal et al. Brain Tumor Classification Using ResNet-101 Based Squeeze and Excitation Deep Neural Network
#     https://ieeexplore.ieee.org/document/8882973

#     """

#     def __init__(
#         self, input_size=[1, 169, 208, 179], gpu=True, output_size=2, dropout=0.5
#     ):
#         model = SECNNDesigner3D()

#         convolutions = nn.Sequential(
#             model.layer0, model.layer1, model.layer2, model.layer3, model.layer4
#         )

#         fc = model.fc

#         super().__init__(
#             convolutions=convolutions,
#             fc=fc,
#             n_classes=output_size,
#             gpu=gpu,
#         )

#     @staticmethod
#     def get_input_size():
#         return "1@169x207x179"

#     @staticmethod
#     def get_dimension():
#         return "3D"

#     @staticmethod
#     def get_task():
#         return ["classification"]


# class Conv5_FC3_SSDA(CNN_SSDA):
#     """
#     Reduce the 2D or 3D input image to an array of size output_size.
#     """

#     def __init__(self, input_size, gpu=True, output_size=2, dropout=0.5):
#         conv, norm, pool = get_layers_fn(input_size)
#         # fmt: off
#         convolutions = nn.Sequential(
#             conv(input_size[0], 8, 3, padding=1),
#             norm(8),
#             nn.ReLU(),
#             pool(2, 2),

#             conv(8, 16, 3, padding=1),
#             norm(16),
#             nn.ReLU(),
#             pool(2, 2),

#             conv(16, 32, 3, padding=1),
#             norm(32),
#             nn.ReLU(),
#             pool(2, 2),

#             conv(32, 64, 3, padding=1),
#             norm(64),
#             nn.ReLU(),
#             pool(2, 2),

#             conv(64, 128, 3, padding=1),
#             norm(128),
#             nn.ReLU(),
#             pool(2, 2),

#             # conv(128, 256, 3, padding=1),
#             # norm(256),
#             # nn.ReLU(),
#             # pool(2, 2),
#         )

#         # Compute the size of the first FC layer
#         input_tensor = torch.zeros(input_size).unsqueeze(0)
#         output_convolutions = convolutions(input_tensor)

#         fc_class_source = nn.Sequential(
#             nn.Flatten(),
#             nn.Dropout(p=dropout),

#             nn.Linear(np.prod(list(output_convolutions.shape)).item(), 1300),
#             nn.ReLU(),

#             nn.Linear(1300, 50),
#             nn.ReLU(),

#             nn.Linear(50, output_size)
#         )


#         fc_class_target= nn.Sequential(
#             nn.Flatten(),
#             nn.Dropout(p=dropout),

#             nn.Linear(np.prod(list(output_convolutions.shape)).item(), 1300),
#             nn.ReLU(),

#             nn.Linear(1300, 50),
#             nn.ReLU(),

#             nn.Linear(50, output_size)
#         )

#         fc_domain = nn.Sequential(
#             nn.Flatten(),
#             nn.Dropout(p=dropout),

#             nn.Linear(np.prod(list(output_convolutions.shape)).item(), 1300),
#             nn.ReLU(),

#             nn.Linear(1300, 50),
#             nn.ReLU(),

#             nn.Linear(50, output_size)
#         )
#         # fmt: on
#         super().__init__(
#             convolutions=convolutions,
#             fc_class_source=fc_class_source,
#             fc_class_target=fc_class_target,
#             fc_domain=fc_domain,
#             n_classes=output_size,
#             gpu=gpu,
#         )

#     @staticmethod
#     def get_input_size():
#         return "1@128x128"

#     @staticmethod
#     def get_dimension():
#         return "2D or 3D"

#     @staticmethod
#     def get_task():
#         return ["classification", "regression"]
