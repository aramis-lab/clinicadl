import numpy as np
import torch
import torch.utils.model_zoo as model_zoo
from torch import nn
from torchvision.models.resnet import BasicBlock

from clinicadl.utils.network.cnn.resnet import ResNetDesigner, model_urls
from clinicadl.utils.network.cnn.resnet3D import ResNetDesigner3D
from clinicadl.utils.network.cnn.SECNN import SECNNDesigner3D
from clinicadl.utils.network.network_utils import PadMaxPool2d, PadMaxPool3d
from clinicadl.utils.network.sub_network import CNN, CNN_SSDA


def get_layers_fn(input_size):
    if len(input_size) == 4:
        return nn.Conv3d, nn.BatchNorm3d, PadMaxPool3d
    elif len(input_size) == 3:
        return nn.Conv2d, nn.BatchNorm2d, PadMaxPool2d
    else:
        raise ValueError(
            f"The input is neither a 2D or 3D image.\n "
            f"Input shape is {input_size - 1}."
        )


class Conv5_FC3(CNN):
    """
    It is a convolutional neural network with 5 convolution and 3 fully-connected layer.
    It reduces the 2D or 3D input image to an array of size output_size.
    """

    def __init__(self, input_size, gpu=True, output_size=2, dropout=0.5):
        conv, norm, pool = get_layers_fn(input_size)
        # fmt: off
        convolutions = nn.Sequential(
            conv(input_size[0], 8, 3, padding=1),
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

        # Compute the size of the first FC layer
        input_tensor = torch.zeros(input_size).unsqueeze(0)
        output_convolutions = convolutions(input_tensor)

        fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),

            nn.Linear(np.prod(list(output_convolutions.shape)).item(), 1300),
            nn.ReLU(),

            nn.Linear(1300, 50),
            nn.ReLU(),

            nn.Linear(50, output_size)
        )
        # fmt: on
        super().__init__(
            convolutions=convolutions,
            fc=fc,
            n_classes=output_size,
            gpu=gpu,
        )

    @staticmethod
    def get_input_size():
        return "1@128x128"

    @staticmethod
    def get_dimension():
        return "2D or 3D"

    @staticmethod
    def get_task():
        return ["classification", "regression"]


class Conv4_FC3(CNN):
    """
    Convolutional neural network with 4 convolution and 3 fully-connected layer.
    Reduce the 2D or 3D input image to an array of size output_size.
    """

    def __init__(self, input_size, gpu=True, output_size=2, dropout=0.5):
        conv, norm, pool = get_layers_fn(input_size)
        # fmt: off
        convolutions = nn.Sequential(
            conv(input_size[0], 8, 3, padding=1),
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

        # Compute the size of the first FC layer
        input_tensor = torch.zeros(input_size).unsqueeze(0)
        output_convolutions = convolutions(input_tensor)

        fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),

            nn.Linear(np.prod(list(output_convolutions.shape)).item(), 50),
            nn.ReLU(),

            nn.Linear(50, 40),
            nn.ReLU(),

            nn.Linear(40, output_size)
        )
        # fmt: on
        super().__init__(
            convolutions=convolutions,
            fc=fc,
            n_classes=output_size,
            gpu=gpu,
        )

    @staticmethod
    def get_input_size():
        return "1@128x128"

    @staticmethod
    def get_dimension():
        return "2D or 3D"

    @staticmethod
    def get_task():
        return ["classification", "regression"]


class resnet18(CNN):
    """
    ResNet-18 is a neural network that is 18 layers deep based on residual block.
    It uses skip connections or shortcuts to jump over some layers.
    It is an image classification pre-trained model.
    The model input has 3 channels in RGB order.

    Reference: Kaiming He et al., Deep Residual Learning for Image Recognition.
    https://arxiv.org/abs/1512.03385?context=cs
    """

    def __init__(self, input_size, gpu=False, output_size=2, dropout=0.5):
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

        super().__init__(
            convolutions=convolutions,
            fc=fc,
            n_classes=output_size,
            gpu=gpu,
        )

    @staticmethod
    def get_input_size():
        return "3@128x128"

    @staticmethod
    def get_dimension():
        return "2D"

    @staticmethod
    def get_task():
        return ["classification"]


class Stride_Conv5_FC3(CNN):
    """
    Convolutional neural network with 5 convolution and 3 fully-connected layer and a stride of 2 for each convolutional layer.
    Reduce the 2D or 3D input image to an array of size output_size.
    """

    def __init__(self, input_size, gpu=True, output_size=2, dropout=0.5):
        conv, norm, pool = get_layers_fn(input_size)
        # fmt: off
        convolutions = nn.Sequential(
            conv(input_size[0], 8, 3, padding=1, stride=2),
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

        # Compute the size of the first FC layer
        input_tensor = torch.zeros(input_size).unsqueeze(0)
        output_convolutions = convolutions(input_tensor)

        fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),

            nn.Linear(np.prod(list(output_convolutions.shape)).item(), 1300),
            nn.ReLU(),

            nn.Linear(1300, 50),
            nn.ReLU(),

            nn.Linear(50, output_size)
        )
        # fmt: on
        super().__init__(
            convolutions=convolutions,
            fc=fc,
            n_classes=output_size,
            gpu=gpu,
        )

    @staticmethod
    def get_input_size():
        return "1@128x128"

    @staticmethod
    def get_dimension():
        return "2D or 3D"

    @staticmethod
    def get_task():
        return ["classification", "regression"]


class ResNet3D(CNN):
    """
    ResNet3D is a 3D neural network composed of 5 residual blocks. Each residual block
    is compose of 3D convolutions followed by a batch normalization and an activation function.
    It uses skip connections or shortcuts to jump over some layers. It's a 3D version of the
    original implementation of Kaiming He et al.

    Reference: Kaiming He et al., Deep Residual Learning for Image Recognition.
    https://arxiv.org/abs/1512.03385?context=cs
    """

    def __init__(
        self, input_size=[1, 169, 208, 179], gpu=False, output_size=2, dropout=0.5
    ):
        model = ResNetDesigner3D()

        convolutions = nn.Sequential(
            model.layer0, model.layer1, model.layer2, model.layer3, model.layer4
        )

        fc = model.fc

        super().__init__(
            convolutions=convolutions,
            fc=fc,
            n_classes=output_size,
            gpu=gpu,
        )

    @staticmethod
    def get_input_size():
        return "1@169x207x179"

    @staticmethod
    def get_dimension():
        return "3D"

    @staticmethod
    def get_task():
        return ["classification", "regression"]


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

    def __init__(
        self, input_size=[1, 169, 208, 179], gpu=True, output_size=2, dropout=0.5
    ):
        model = SECNNDesigner3D()

        convolutions = nn.Sequential(
            model.layer0, model.layer1, model.layer2, model.layer3, model.layer4
        )

        fc = model.fc

        super().__init__(
            convolutions=convolutions,
            fc=fc,
            n_classes=output_size,
            gpu=gpu,
        )

    @staticmethod
    def get_input_size():
        return "1@169x207x179"

    @staticmethod
    def get_dimension():
        return "3D"

    @staticmethod
    def get_task():
        return ["classification"]


class Conv5_FC3_SSDA(CNN_SSDA):
    """
    Reduce the 2D or 3D input image to an array of size output_size.
    """

    def __init__(self, input_size, gpu=True, output_size=2, dropout=0.5):
        conv, norm, pool = get_layers_fn(input_size)
        # fmt: off
        convolutions = nn.Sequential(
            conv(input_size[0], 8, 3, padding=1),
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

            # conv(128, 256, 3, padding=1),
            # norm(256),
            # nn.ReLU(),
            # pool(2, 2),
        )

        # Compute the size of the first FC layer
        input_tensor = torch.zeros(input_size).unsqueeze(0)
        output_convolutions = convolutions(input_tensor)

        fc_class_source = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),

            nn.Linear(np.prod(list(output_convolutions.shape)).item(), 1300),
            nn.ReLU(),

            nn.Linear(1300, 50),
            nn.ReLU(),

            nn.Linear(50, output_size)
        )


        fc_class_target= nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),

            nn.Linear(np.prod(list(output_convolutions.shape)).item(), 1300),
            nn.ReLU(),

            nn.Linear(1300, 50),
            nn.ReLU(),

            nn.Linear(50, output_size)
        )

        fc_domain = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),

            nn.Linear(np.prod(list(output_convolutions.shape)).item(), 1300),
            nn.ReLU(),

            nn.Linear(1300, 50),
            nn.ReLU(),

            nn.Linear(50, output_size)
        )
        # fmt: on
        super().__init__(
            convolutions=convolutions,
            fc_class_source=fc_class_source,
            fc_class_target=fc_class_target,
            fc_domain=fc_domain,
            n_classes=output_size,
            gpu=gpu,
        )

    @staticmethod
    def get_input_size():
        return "1@128x128"

    @staticmethod
    def get_dimension():
        return "2D or 3D"

    @staticmethod
    def get_task():
        return ["classification", "regression"]
